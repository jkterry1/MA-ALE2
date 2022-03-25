from all.agents.ppo import PPO

class PPONFSPAgent(PPO):

    def __init__(self, features, v, policy):
        super().__init__(features, v, policy)
