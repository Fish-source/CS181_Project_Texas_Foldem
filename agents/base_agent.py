from abc import ABC, abstractmethod
from typing import Dict, List

from config import Action


class BaseAgent(ABC):

    @abstractmethod
    def act(self, observation: Dict, legal_actions: List[Action]) -> Action:
        raise NotImplementedError

    def reset(self):
        pass

    def observe_transition(self, state, action, reward, next_state, done):
        pass
