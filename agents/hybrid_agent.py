import random

from agents.base_agent import BaseAgent
from agents.gto_agent import GTOAgent
from agents.random_agent import RandomAgent
from config import Action


class HybridAgent(BaseAgent):

    def __init__(self, gto_ratio=0.3, gto_epsilon=0.1,
                 gto_num_simulations=500, seed=None):
        self.gto_ratio = gto_ratio
        self.gto = GTOAgent(epsilon=gto_epsilon,
                            num_simulations=gto_num_simulations)
        self.random_agent = RandomAgent()
        self._rng = random.Random(seed)

    def act(self, observation, legal_actions):
        if not legal_actions:
            return Action.FOLD
        if self._rng.random() < self.gto_ratio:
            return self.gto.act(observation, legal_actions)
        return self.random_agent.act(observation, legal_actions)

    def reset(self):
        self.gto.reset()
        self.random_agent.reset()

    @property
    def name(self):
        return 'Hybrid(GTO=%d%%)' % int(self.gto_ratio * 100)
