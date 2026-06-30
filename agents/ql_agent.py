import pickle
from collections import defaultdict

from agents.base_agent import BaseAgent
from config import (
    Action, QL_LEARNING_RATE, QL_DISCOUNT_FACTOR,
    QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
)
from training.state_encoder import StateEncoder


class QLAgent(BaseAgent):

    def __init__(
        self,
        state_encoder=None,
        learning_rate=QL_LEARNING_RATE,
        discount_factor=QL_DISCOUNT_FACTOR,
        epsilon=QL_EPSILON_START,
        epsilon_min=QL_EPSILON_MIN,
        epsilon_decay=QL_EPSILON_DECAY,
        training=False,
    ):
        self.state_encoder = state_encoder or StateEncoder(training=training)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training = training
        self.q_table = defaultdict(lambda: defaultdict(float))
        self._last_state = None
        self._last_action = None

    def act(self, observation, legal_actions):
        if not legal_actions:
            return Action.FOLD

        import random
        if self.training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_key = self.state_encoder.encode(observation)
        q_values = self.q_table[state_key]

        best_action = legal_actions[0]
        best_value = float('-inf')
        for action in legal_actions:
            val = q_values[action]
            if val > best_value:
                best_value = val
                best_action = action

        return best_action

    def observe_transition(self, state, action, reward, next_state, done):
        if not self.training:
            return

        state_key = self.state_encoder.encode(state)
        next_state_key = self.state_encoder.encode(next_state) if not done else None

        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            legal_next = next_state.get('legal_actions', [])
            if legal_next:
                max_next_q = max(self.q_table[next_state_key][a] for a in legal_next)
            else:
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q

        self.q_table[state_key][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def reset(self):
        self._last_state = None
        self._last_action = None

    def set_training(self, mode=True):
        self.training = mode
        self.state_encoder.training = mode

    def save(self, path):
        serializable = {
            k: dict(v) for k, v in self.q_table.items()
        }
        with open(path, 'wb') as f:
            pickle.dump(serializable, f)

    def load(self, path):
        with open(path, 'rb') as f:
            serializable = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in serializable.items():
            for action, val in v.items():
                self.q_table[k][action] = val
