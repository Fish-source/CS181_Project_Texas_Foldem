import argparse

from agents.random_agent import RandomAgent
from agents.ql_agent import QLAgent
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


def cmd_train(args):
    agent = QLAgent(training=True)
    if args.load:
        agent.load(args.load)
    trainer = Trainer(
        agent=agent,
        num_episodes=args.episodes,
        self_play_ratio=args.self_play,
        eval_interval=args.eval_interval,
        save_path=args.save,
    )
    rewards, win_rates = trainer.train()
    print('\nFinal evaluation...')
    trainer.evaluate(num_games=args.eval_games)


def cmd_eval(args):
    agent = QLAgent(training=False)
    if args.load:
        agent.load(args.load)
    agent.set_training(False)
    agent.epsilon = 0.0

    evaluator = Evaluator()
    random_agent = RandomAgent()

    result = evaluator.run_match(agent, random_agent, num_games=args.games)
    print('Results vs RandomAgent (%d games):' % args.games)
    print('  Win rate: %.2f%%' % (result['win_rate_a'] * 100))
    print('  Avg payoff: %.2f' % result['avg_reward_a'])
    print('  Action entropy: %.3f' % result['action_entropy_a'])
    print('  Action distribution: %s' % {
        'FOLD': result['actions_a'].get(0, 0),
        'CALL': result['actions_a'].get(1, 0),
        'RAISE_HALF': result['actions_a'].get(2, 0),
        'RAISE_POT': result['actions_a'].get(3, 0),
    })


def cmd_tournament(args):
    agent = QLAgent(training=False)
    if args.load:
        agent.load(args.load)
    agent.set_training(False)
    agent.epsilon = 0.0

    random_agent = RandomAgent()
    evaluator = Evaluator()

    agents = [agent, random_agent]
    names = ['QLearning', 'Random']
    results = evaluator.run_tournament(agents, names, num_games=args.games)


def main():
    parser = argparse.ArgumentParser(description='Texas Foldem - Poker AI Research')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help='Train Q-learning agent')
    train_parser.add_argument('--episodes', type=int, default=10000)
    train_parser.add_argument('--self-play', type=float, default=0.7, dest='self_play')
    train_parser.add_argument('--eval-interval', type=int, default=1000, dest='eval_interval')
    train_parser.add_argument('--eval-games', type=int, default=500, dest='eval_games')
    train_parser.add_argument('--save', type=str, default='q_table.pkl')
    train_parser.add_argument('--load', type=str, default=None)

    eval_parser = subparsers.add_parser('eval', help='Evaluate agent vs random')
    eval_parser.add_argument('--load', type=str, default='q_table.pkl')
    eval_parser.add_argument('--games', type=int, default=1000)

    tourney_parser = subparsers.add_parser('tournament', help='Run tournament')
    tourney_parser.add_argument('--load', type=str, default='q_table.pkl')
    tourney_parser.add_argument('--games', type=int, default=1000)

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'tournament':
        cmd_tournament(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
