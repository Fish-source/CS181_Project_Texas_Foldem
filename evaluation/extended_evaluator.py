import math
from collections import defaultdict
from typing import Dict, List, Optional

from engine.poker_env import PokerEnv
from config import Action, Stage, INITIAL_CHIPS


class ExtendedEvaluator:

    def __init__(self):
        self.env = PokerEnv()

    def run_match_detailed(self, agent_a, agent_b, num_games=1000):
        wins_a = 0
        wins_b = 0
        ties = 0
        total_payoff_a = 0
        total_payoff_b = 0
        action_counts_a = defaultdict(int)
        action_counts_b = defaultdict(int)
        stage_action_a = {s: defaultdict(int) for s in Stage}
        stage_action_b = {s: defaultdict(int) for s in Stage}
        pot_sizes = []
        game_lengths = []
        rolling_wins = []
        showdown_count = 0
        fold_win_a = 0
        fold_win_b = 0
        chip_ev_a = []

        for gi in range(num_games):
            obs = self.env.reset()
            agent_a.reset()
            agent_b.reset()
            done = False
            steps = 0

            while not done:
                current = self.env.current_player
                legal = obs['legal_actions']
                stage = obs.get('stage', Stage.PREFLOP)

                if current == 0:
                    action = agent_a.act(obs, legal)
                    action_counts_a[action] += 1
                    stage_action_a[stage][action] += 1
                else:
                    action = agent_b.act(obs, legal)
                    action_counts_b[action] += 1
                    stage_action_b[stage][action] += 1

                obs, reward, done, info = self.env.step(action)
                steps += 1

            game_lengths.append(steps)
            if done and 'payoffs' in info:
                pa, pb = info['payoffs']
                total_payoff_a += pa
                total_payoff_b += pb
                chip_ev_a.append(pa)
                pot_sizes.append(self.env.state.pot if self.env.state else 0)
                if pa > pb:
                    wins_a += 1
                    rolling_wins.append(1)
                    if info.get('winner') == 0 and self.env.state and self.env.state.players[1].is_folded:
                        fold_win_a += 1
                elif pb > pa:
                    wins_b += 1
                    rolling_wins.append(0)
                    if info.get('winner') == 1 and self.env.state and self.env.state.players[0].is_folded:
                        fold_win_b += 1
                else:
                    ties += 1
                    rolling_wins.append(0.5)

                if self.env.state and not self.env.state.players[0].is_folded and not self.env.state.players[1].is_folded:
                    showdown_count += 1

        n = num_games
        return {
            'win_rate_a': wins_a / n,
            'win_rate_b': wins_b / n,
            'tie_rate': ties / n,
            'avg_reward_a': total_payoff_a / n,
            'avg_reward_b': total_payoff_b / n,
            'action_entropy_a': self._action_entropy(action_counts_a),
            'action_entropy_b': self._action_entropy(action_counts_b),
            'actions_a': dict(action_counts_a),
            'actions_b': dict(action_counts_b),
            'stage_actions_a': {s: dict(v) for s, v in stage_action_a.items()},
            'stage_actions_b': {s: dict(v) for s, v in stage_action_b.items()},
            'avg_pot': sum(pot_sizes) / len(pot_sizes) if pot_sizes else 0,
            'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
            'rolling_wins': rolling_wins,
            'showdown_rate': showdown_count / n,
            'fold_win_a': fold_win_a,
            'fold_win_b': fold_win_b,
            'chip_ev_a': chip_ev_a,
            'pot_sizes': pot_sizes,
        }

    def run_tournament_detailed(self, agents, agent_names, num_games=500):
        n = len(agents)
        results = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = '%s vs %s' % (agent_names[i], agent_names[j])
                result = self.run_match_detailed(agents[i], agents[j], num_games)
                results[key] = result
                print('%s: win=%.2f%% avg_payoff=%.2f' % (
                    key, result['win_rate_a'] * 100, result['avg_reward_a']))
        return results

    def compute_rolling_win_rate(self, rolling_wins, window=100):
        rates = []
        for i in range(len(rolling_wins)):
            start = max(0, i - window + 1)
            segment = rolling_wins[start:i + 1]
            rates.append(sum(segment) / len(segment))
        return rates

    def compute_agent_profile(self, result, agent_side='a'):
        prefix = agent_side
        actions = result.get('actions_%s' % prefix, {})
        stage_actions = result.get('stage_actions_%s' % prefix, {})
        total = sum(actions.values()) or 1

        fold_rate = actions.get(Action.FOLD, 0) / total
        call_rate = actions.get(Action.CALL, 0) / total
        raise_half_rate = actions.get(Action.RAISE_HALF_POT, 0) / total
        raise_pot_rate = actions.get(Action.RAISE_POT, 0) / total
        aggression = (raise_half_rate + raise_pot_rate)

        preflop_actions = stage_actions.get(Stage.PREFLOP, {})
        preflop_total = sum(preflop_actions.values()) or 1
        preflop_raise = (preflop_actions.get(Action.RAISE_HALF_POT, 0) +
                         preflop_actions.get(Action.RAISE_POT, 0)) / preflop_total

        river_actions = stage_actions.get(Stage.RIVER, {})
        river_total = sum(river_actions.values()) or 1
        river_fold = river_actions.get(Action.FOLD, 0) / river_total

        return {
            'fold_rate': fold_rate,
            'call_rate': call_rate,
            'aggression': aggression,
            'preflop_raise_rate': preflop_raise,
            'river_fold_rate': river_fold,
            'action_entropy': result.get('action_entropy_%s' % prefix, 0),
            'win_rate': result.get('win_rate_%s' % prefix, 0),
            'avg_reward': result.get('avg_reward_%s' % prefix, 0),
            'showdown_rate': result.get('showdown_rate', 0),
        }

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
