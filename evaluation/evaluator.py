import math
from collections import defaultdict

from engine.poker_env import PokerEnv
from config import Action, INITIAL_CHIPS


class Evaluator:

    def __init__(self):
        self.env = PokerEnv()

    def run_match(self, agent_a, agent_b, num_games=1000):
        wins_a = 0
        wins_b = 0
        ties = 0
        total_payoff_a = 0
        total_payoff_b = 0
        action_counts_a = defaultdict(int)
        action_counts_b = defaultdict(int)

        for _ in range(num_games):
            obs = self.env.reset()
            agent_a.reset()
            agent_b.reset()
            done = False

            while not done:
                current = self.env.current_player
                legal = obs['legal_actions']

                if current == 0:
                    action = agent_a.act(obs, legal)
                    action_counts_a[action] += 1
                else:
                    action = agent_b.act(obs, legal)
                    action_counts_b[action] += 1

                obs, reward, done, info = self.env.step(action)

            if done and 'payoffs' in info:
                pa, pb = info['payoffs']
                total_payoff_a += pa
                total_payoff_b += pb
                if pa > pb:
                    wins_a += 1
                elif pb > pa:
                    wins_b += 1
                else:
                    ties += 1

        n = num_games
        entropy_a = self._action_entropy(action_counts_a)
        entropy_b = self._action_entropy(action_counts_b)

        return {
            'win_rate_a': wins_a / n,
            'win_rate_b': wins_b / n,
            'tie_rate': ties / n,
            'avg_reward_a': total_payoff_a / n,
            'avg_reward_b': total_payoff_b / n,
            'action_entropy_a': entropy_a,
            'action_entropy_b': entropy_b,
            'actions_a': dict(action_counts_a),
            'actions_b': dict(action_counts_b),
        }

    def run_tournament(self, agents, agent_names, num_games=1000):
        n = len(agents)
        results = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = '%s vs %s' % (agent_names[i], agent_names[j])
                result = self.run_match(agents[i], agents[j], num_games)
                results[key] = result
                print('%s: win=%.2f%% avg_payoff=%.2f' % (
                    key,
                    result['win_rate_a'] * 100,
                    result['avg_reward_a'],
                ))

        return results

    @staticmethod
    def _action_entropy(action_counts):
        total = sum(action_counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in action_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
