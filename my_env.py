from itertools import cycle
from all.environments import MultiagentAtariEnv, MultiagentPettingZooEnv


class MAPZEnvSteps(MultiagentPettingZooEnv):
    """`MultiagentPettingZooEnv` that includes the current num
    steps within current episode in the returned state dict.
    This is necessary for NFSP to determine which policy to use for a
    given episode.
    """
    def __init__(self, zoo_env, name, device='cuda'):
        MultiagentPettingZooEnv.__init__(self, zoo_env, name, device=device)
        # self._episodes_seen = -1 # incremented on reset(), start at -1
        self._ep_steps = None


    def _add_env_steps(self, state):
        cur_agent = state['agent']
        state['ep_step'] = self._ep_steps[cur_agent]
        return state

    def reset(self):
        # self._episodes_seen += 1
        self._ep_steps = {ag: 0 for ag in self.agents}
        self._agent_looper = cycle(self.agents)
        return super(MAPZEnvSteps, self).reset()
        # return self._add_env_steps(state)

    def step(self, action):
        self._ep_steps[next(self._agent_looper)] += 1
        return super(MAPZEnvSteps, self).step(action)
        # return self._add_env_steps(state)

    def last(self):
        state = super(MAPZEnvSteps, self).last()
        return self._add_env_steps(state)
