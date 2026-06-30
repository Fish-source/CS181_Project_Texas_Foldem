import random
from tqdm import tqdm

from engine.poker_env import PokerEnv
from agents.random_agent import RandomAgent
from agents.ql_agent import QLAgent
from config import Action, INITIAL_CHIPS


class Trainer:

    def __init__(
        self,
        agent=None,
        num_episodes=10000,
        self_play_ratio=0.7,
        eval_interval=1000,
        eval_games=500,
        save_path='q_table.pkl',
    ):
        self.env = PokerEnv()
        self.agent = agent or QLAgent(training=True)
        self.agent.set_training(True)
        self.num_episodes = num_episodes
        self.self_play_ratio = self_play_ratio
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.save_path = save_path
        self.random_agent = RandomAgent()
        self.rewards_history = []
        self.win_rates_history = []

    def _play_one_game(self, opponent):
        obs = self.env.reset()
        self.agent.reset()
        opponent.reset()

        pending = []
        done = False

        while not done:
            current = self.env.current_player
            legal = obs['legal_actions']

            if current == 0:
                action = self.agent.act(obs, legal)
                pending.append((obs, action))
            else:
                action = opponent.act(obs, legal)

            next_obs, reward, done, info = self.env.step(action)

            if current == 0 and pending:
                prev_obs, prev_action = pending.pop()
                terminal_reward = 0.0
                if done and 'payoffs' in info:
                    terminal_reward = float(info['payoffs'][0]) / INITIAL_CHIPS
                self.agent.observe_transition(prev_obs, prev_action, terminal_reward, next_obs, done)

            obs = next_obs

        self.agent.decay_epsilon()

        if done and 'payoffs' in info:
            return info['payoffs'][0]
        return 0

    def train(self):
        wins = 0
        total_reward = 0

        pbar = tqdm(range(self.num_episodes), desc='Training')
        for ep in pbar:
            if random.random() < self.self_play_ratio:
                opponent = self.agent
            else:
                opponent = self.random_agent

            payoff = self._play_one_game(opponent)
            total_reward += payoff
            if payoff > 0:
                wins += 1

            self.rewards_history.append(payoff)

            if (ep + 1) % self.eval_interval == 0:
                recent = self.rewards_history[-self.eval_interval:]
                avg_reward = sum(recent) / len(recent)
                win_rate = wins / (ep + 1)
                self.win_rates_history.append(win_rate)
                pbar.set_postfix({
                    'eps': '%.3f' % self.agent.epsilon,
                    'avg_r': '%.1f' % avg_reward,
                    'win': '%.2f' % win_rate,
                    'Q_size': len(self.agent.q_table),
                })
                wins = 0
                total_reward = 0

        self.agent.save(self.save_path)
        print('Training complete. Q-table saved to %s (%d states)' % (self.save_path, len(self.agent.q_table)))
        return self.rewards_history, self.win_rates_history

    def evaluate(self, num_games=1000):
        self.agent.set_training(False)
        self.agent.epsilon = 0.0

        wins = 0
        total_payoff = 0
        action_counts = {a: 0 for a in Action}

        for _ in range(num_games):
            obs = self.env.reset()
            self.agent.reset()
            self.random_agent.reset()
            done = False

            while not done:
                current = self.env.current_player
                legal = obs['legal_actions']

                if current == 0:
                    action = self.agent.act(obs, legal)
                    action_counts[action] += 1
                else:
                    action = self.random_agent.act(obs, legal)

                obs, reward, done, info = self.env.step(action)

            if done and 'payoffs' in info:
                payoff = info['payoffs'][0]
                total_payoff += payoff
                if payoff > 0:
                    wins += 1

        win_rate = wins / num_games
        avg_payoff = total_payoff / num_games

        print('Evaluation vs Random (%d games):' % num_games)
        print('  Win rate: %.2f%%' % (win_rate * 100))
        print('  Avg payoff: %.2f' % avg_payoff)
        print('  Action distribution: %s' % {Action(a).name: c for a, c in action_counts.items() if c > 0})

        self.agent.set_training(True)
        return {'win_rate': win_rate, 'avg_payoff': avg_payoff, 'action_counts': action_counts}
