import random
from typing import List, Optional

from treys import Evaluator as TreysEvaluator

from config import (
    HAND_STRENGTH_BUCKETS_RL,
    HAND_STRENGTH_BUCKETS_SIMPLE,
    BOARD_STRENGTH_BUCKETS,
    MONTE_CARLO_SIMULATIONS_DEFAULT,
)
from engine.card import Card, ALL_CARDS, Suit, Rank


_treys_eval = TreysEvaluator()


class HandEvaluator:
    def evaluate(self, hand: List[Card], board: List[Card]) -> int:
        if not hand or len(board) < 3:
            return 7463
        treys_hand = [c.treys_card for c in hand]
        treys_board = [c.treys_card for c in board]
        return _treys_eval.evaluate(treys_board, treys_hand)

    def monte_carlo_win_rate(
        self,
        hand: List[Card],
        board: List[Card],
        num_simulations=MONTE_CARLO_SIMULATIONS_DEFAULT,
    ) -> float:
        if len(hand) != 2:
            return 0.5

        known = set((c.suit, c.rank) for c in hand + board)
        remaining = [c for c in ALL_CARDS if (c.suit, c.rank) not in known]
        cards_needed_on_board = 5 - len(board)
        wins = 0
        ties = 0

        for _ in range(num_simulations):
            random.shuffle(remaining)
            opp_hand = remaining[:2]
            sim_board_cards = remaining[2:2 + cards_needed_on_board]
            full_board = board + sim_board_cards

            my_rank = self.evaluate(hand, full_board)
            opp_rank = self.evaluate(opp_hand, full_board)

            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                ties += 1

        return (wins + 0.5 * ties) / num_simulations

    def hand_strength_bucket(
        self,
        win_rate: float,
        n_buckets: int = HAND_STRENGTH_BUCKETS_SIMPLE,
    ) -> int:
        bucket = int(win_rate * n_buckets)
        return min(bucket, n_buckets - 1)

    def board_strength_bucket(self, hand: List[Card], board: List[Card]) -> int:
        if len(board) < 3:
            return 0

        hand_rank = self.evaluate(hand, board)
        rank_class = _treys_eval.get_rank_class(hand_rank)
        if rank_class <= 2:
            return BOARD_STRENGTH_BUCKETS - 1
        elif rank_class <= 4:
            return BOARD_STRENGTH_BUCKETS - 2
        elif rank_class <= 6:
            return BOARD_STRENGTH_BUCKETS - 3
        elif rank_class <= 8:
            return BOARD_STRENGTH_BUCKETS - 4
        else:
            return 0
