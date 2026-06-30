import random

from agents.base_agent import BaseAgent
from config import Action


class RandomAgent(BaseAgent):

    def act(self, observation, legal_actions):
        if not legal_actions:
            return Action.FOLD
        return random.choice(legal_actions)
