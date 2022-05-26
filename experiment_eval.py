import argparse
import torch
torch.set_num_threads(1)
import os
import numpy as np
import random
import subprocess
import shutil
from PIL import Image
from datetime import datetime
from env_utils import make_env, InvertColorAgentIndicator, CropObservation
from all.agents.independent import IndependentMultiagent

from algorithms.shared_rainbow import make_rainbow_preset
from algorithms.independent_rainbow import make_indepedent_rainbow
# from ppo_ram import make_ppo_ram_vec
from algorithms.shared_ppo import make_ppo_vec, make_ppo_vec_largenet
from algorithms.rainbow_nfsp import make_nfsp_rainbow

trainer_types = {
    "shared_rainbow": make_rainbow_preset,
    "independent_rainbow": make_indepedent_rainbow,
    "shared_ppo": make_ppo_vec,
    # "shared_ppo_ram": make_ppo_ram_vec,
    "shared_ppo_largenet": make_ppo_vec_largenet,
    "nfsp_rainbow": make_nfsp_rainbow,
}

class TestRandom:
    def __init__(self):
        pass
    def act(self, state):
        return random.randint(0,17)


def generate_episode_gifs(env, _agent, max_frames, dir, cropped=False):
    # initialize the episode
    state = env.reset()
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    for agent in env.agent_iter():
        action = _agent.act(state) if not state.done else None
        state = env.step(action)
        obs = env._env.unwrapped.ale.getScreenRGB()
        if cropped:
            obs = obs[CropObservation.DEFAULT_CROP_INDEX.get(env.name, ())]
        if not prev_obs or not np.equal(obs, prev_obs).all():
            im = Image.fromarray(obs)
            im.save(f"{dir}{str(frame_idx).zfill(4)}.png")

            frame_idx += 1
            if frame_idx >= max_frames:
                break

        if len(env._env.agents) == 1 and env._env.dones[env._env.agent_selection]:
            break

def test_single_episode(env, _agent, generate_gif_callback=None):
    # initialize the episode
    state = env.reset()
    returns = {agent: 0 for agent in env.agents}
    num_steps = 0
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    for agent in env.agent_iter():
        action = _agent.act(state) if not state.done else None
        state = env.step(action)
        if state is None:
            break
        returns[env._env.agent_selection] += state.reward
        num_steps += 1

        if len(env._env.agents) == 1 and env._env.dones[env._env.agent_selection]:
            break

    print(returns)
    return returns, num_steps

def test_independent(env, agent, frames):
    returns = []
    num_steps = 0
    while num_steps < frames:
        episode_return, ep_steps = test_single_episode(env, agent)
        returns.append(episode_return)
        num_steps += ep_steps
        # self._log_test_episode(episode, episode_return)
    return returns

def returns_agent(returns, agent):
    if agent not in returns[0]:
        return np.float("nan")
    agent_1_returns = [ret[agent] for ret in returns]
    return np.mean(agent_1_returns)


from all.agents import Agent
from all.core import State
import torch
from env_utils import MAPZEnvSteps

class SingleEnvAgent(Agent):
    def __init__(self, agent):
        self.agent = agent

    def act(self, state):
        return self.agent.act(State.array([state]))


def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument(
        "checkpoint", help="Checkpoint number."
    )
    parser.add_argument(
        "folder", help="Folder with checkpoints."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--vs-random", action="store_true", help="Play first_0 vs random for all other players."
    )
    parser.add_argument(
        "--vs-builtin", action="store_true", help="Play first_0 vs the builtin agent for other player (only compatable with certain environments which have the builtin player)."
    )
    parser.add_argument(
        "--agent-random", action="store_true", help="Play first_0 vs random for all other players."
    )
    parser.add_argument(
        "--frames", type=int, default=100000, help="The number of training frames."
    )
    parser.add_argument(
        "--agent", type=str, default="first_0", help="Agent to print out value."
    )
    parser.add_argument(
        "--generate-gif", action="store_true", help="flag to save rendered gif"
    )
    parser.add_argument(
        "--cropped", action="store_true", help="flag to crop rendered image as agents see."
    )
    args = parser.parse_args()


    checkpoint_path = os.path.join(args.folder,f"{args.checkpoint}.pt")
    print(checkpoint_path)
    frame_skip = 1 if args.generate_gif else 4

    preset = torch.load(checkpoint_path, map_location=args.device)

    base_agent = preset.test_agent()
    env = make_env(args.env, vs_builtin=args.vs_builtin)
    if hasattr(base_agent, 'agents'):
        agent = base_agent
        if args.env.startswith("surround") and args.vs_builtin:
            # in surround builtin, the agent is the equivalent of the
            # second agent in the
            agent = IndependentMultiagent({
                "first_0" : agent.agents['second_0']
            })
    else:
        agent = SingleEnvAgent(preset.test_agent())
        agent = IndependentMultiagent({
            agent_id : agent
            for agent_id in env.possible_agents
        })
        env = InvertColorAgentIndicator(env)
    # env = MultiagentPettingZooEnv(env, args.env, device=args.device)
    env = MAPZEnvSteps(env, args.env, device=args.device)
    state = env.reset()
    agent.act(state)


    if args.vs_random:
        for a in env._env.possible_agents:
            if a != args.agent:
                agent.agents[a] = TestRandom()
    if args.agent_random:
        agent.agents[args.agent] = TestRandom()


    if not args.generate_gif:
        returns = test_independent(env, agent, args.frames)
        print(returns)
        agent_names = ["first_0", "second_0", "third_0", "fourth_0"]
        open("out.txt",'a').write(f"{args.folder},{args.checkpoint},{args.agent},{','.join(str(returns_agent(returns, agent)) for agent in agent_names)},{args.vs_random},{args.agent_random}\n")
        
    else:
        opponent = "vs_random" * args.vs_random + "vs_builtin" * args.vs_builtin + "agent_random" * args.agent_random
        cropped = "cropped" * args.cropped
        now = datetime.now().strftime("%m%d%H%M%S")
        ckpt = int(args.checkpoint)
        name = f"{ckpt}-{args.agent}-{opponent}-{cropped}-{now}"
        
        frame_dir = f"{args.folder}/frames/"
        playback_dir = f"{args.folder}/playbacks/"
        os.makedirs(frame_dir,exist_ok=True)
        os.makedirs(playback_dir,exist_ok=True)
        
        generate_episode_gifs(env, agent, args.frames, frame_dir, args.cropped)
        ffmpeg_command_mp4 = [
            "ffmpeg",
            "-framerate", "60",
            "-i", f"{frame_dir}%04d.png",
            "-vcodec", "libx264",
            "-crf", "1",
            "-pix_fmt", "yuv420p",
            f"{playback_dir}{name}.mp4"
        ]

        ffmpeg_command_gif = [
            'ffmpeg',
            '-i', f'{frame_dir}%04d.png',
            f'{playback_dir}{name}.gif']

        subprocess.run(ffmpeg_command_mp4)
        subprocess.run(ffmpeg_command_gif)
        shutil.rmtree(f"{frame_dir}")


if __name__ == "__main__":
    main()
