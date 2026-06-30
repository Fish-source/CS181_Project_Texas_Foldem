from engine.hand_evaluator import HandEvaluator
from engine.rule_engine import RuleEngine
from config import HAND_STRENGTH_BUCKETS_RL


class StateEncoder:

    def __init__(self, hand_evaluator=None, rule_engine=None, training=False):
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.rule_engine = rule_engine or RuleEngine()
        self.training = training

    def encode(self, observation):
        hand = observation.get('hand', [])
        community_cards = observation.get('community_cards', [])
        pot = observation.get('pot', 0)
        stage = observation.get('stage', 0)

        num_sim = 100 if self.training else 1000
        win_rate = self.hand_evaluator.monte_carlo_win_rate(hand, community_cards, num_sim)
        hand_strength = self.hand_evaluator.hand_strength_bucket(win_rate, HAND_STRENGTH_BUCKETS_RL)
        board_strength = self.hand_evaluator.board_strength_bucket(hand, community_cards)
        pot_bucket = self.rule_engine.pot_bucket(pot)

        return (hand_strength, board_strength, pot_bucket, int(stage))
