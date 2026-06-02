from typing import List, Optional, Tuple

from config import (
    Action, Stage, NUM_PLAYERS, SMALL_BLIND, BIG_BLIND,
    MAX_RAISES_PER_STREET, COMMUNITY_CARDS_BY_STAGE,
    POT_BUCKET_BOUNDS, POT_BUCKETS,
)
from engine.card import Card, Deck
from engine.game_state import GameState, PlayerState
from engine.hand_evaluator import HandEvaluator

_hand_eval = HandEvaluator()


class RuleEngine:

    @staticmethod
    def get_legal_actions(state: GameState) -> List[Action]:
        if state.is_terminal:
            return []

        player = state.players[state.current_player]
        if player.is_folded or player.is_all_in:
            return []

        actions = [Action.FOLD]

        opp_bet = max(p.current_bet for i, p in enumerate(state.players) if i != state.current_player)
        call_amount = opp_bet - player.current_bet

        if call_amount == 0:
            actions.append(Action.CALL)
        elif call_amount <= player.chips:
            actions.append(Action.CALL)

        if state.raises_this_street < MAX_RAISES_PER_STREET and player.chips > call_amount:
            pot_after_call = state.pot + call_amount
            for raise_action, multiplier in [(Action.RAISE_HALF_POT, 0.5), (Action.RAISE_POT, 1.0)]:
                raise_amount = int(pot_after_call * multiplier)
                total_needed = call_amount + raise_amount
                if total_needed <= player.chips and raise_amount >= BIG_BLIND:
                    actions.append(raise_action)

        return actions

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        new_state = state.clone()
        player = new_state.players[new_state.current_player]

        if action == Action.FOLD:
            player.is_folded = True
            new_state.is_terminal = True
            new_state.winner = 1 - new_state.current_player

        elif action == Action.CALL:
            opp_bet = max(p.current_bet for i, p in enumerate(new_state.players) if i != new_state.current_player)
            call_amount = min(opp_bet - player.current_bet, player.chips)
            player.chips -= call_amount
            player.current_bet += call_amount
            player.total_bet += call_amount
            new_state.pot += call_amount
            if player.chips == 0:
                player.is_all_in = True

        elif action in (Action.RAISE_HALF_POT, Action.RAISE_POT):
            opp_bet = max(p.current_bet for i, p in enumerate(new_state.players) if i != new_state.current_player)
            call_amount = opp_bet - player.current_bet
            pot_after_call = new_state.pot + call_amount
            multiplier = 0.5 if action == Action.RAISE_HALF_POT else 1.0
            raise_amount = int(pot_after_call * multiplier)
            total_needed = call_amount + raise_amount
            actual_put = min(total_needed, player.chips)
            player.chips -= actual_put
            player.current_bet += actual_put
            player.total_bet += actual_put
            new_state.pot += actual_put
            new_state.raises_this_street += 1
            new_state.last_raiser = new_state.current_player
            if player.chips == 0:
                player.is_all_in = True

        new_state.acted_this_street[new_state.current_player] = True
        new_state.betting_history.append((new_state.current_player, action, player.current_bet))

        if not new_state.is_terminal:
            if RuleEngine._is_betting_round_over(new_state):
                RuleEngine._advance_stage(new_state)
            else:
                RuleEngine._advance_to_next_player(new_state)

        return new_state

    @staticmethod
    def _is_betting_round_over(state: GameState) -> bool:
        active = [i for i, p in enumerate(state.players) if p.is_active]
        if len(active) <= 1:
            return True

        if not all(state.acted_this_street[i] for i in active):
            return False

        bets = [state.players[i].current_bet for i in active]
        return len(set(bets)) <= 1

    @staticmethod
    def _advance_to_next_player(state: GameState):
        n = len(state.players)
        for offset in range(1, n + 1):
            next_p = (state.current_player + offset) % n
            if state.players[next_p].is_active:
                state.current_player = next_p
                return

    @staticmethod
    def _advance_stage(state: GameState):
        for p in state.players:
            p.current_bet = 0
        state.raises_this_street = 0
        state.last_raiser = -1
        state.acted_this_street = [False] * len(state.players)

        next_stage_val = state.stage + 1
        if next_stage_val > Stage.RIVER:
            RuleEngine._showdown(state)
            return

        state.stage = Stage(next_stage_val)

        active = [i for i, p in enumerate(state.players) if p.is_active]
        if len(active) <= 1:
            RuleEngine._showdown(state)
            return

        sb_idx = (state.dealer_idx + 1) % len(state.players)
        for offset in range(len(state.players)):
            idx = (sb_idx + offset) % len(state.players)
            if state.players[idx].is_active:
                state.current_player = idx
                break

    @staticmethod
    def _showdown(state: GameState):
        state.is_terminal = True
        active = [i for i, p in enumerate(state.players) if not p.is_folded]

        if len(active) == 1:
            state.winner = active[0]
            return

        best_rank = None
        winner = None
        for idx in active:
            player = state.players[idx]
            if len(state.community_cards) >= 3 and len(player.hand) == 2:
                rank = _hand_eval.evaluate(player.hand, state.community_cards)
            else:
                rank = 9999

            if best_rank is None or rank < best_rank:
                best_rank = rank
                winner = idx

        state.winner = winner

    @staticmethod
    def compute_payoffs(state: GameState) -> List[int]:
        if state.winner is None:
            return [0] * len(state.players)
        payoffs = [-p.total_bet for p in state.players]
        payoffs[state.winner] += state.pot
        return payoffs

    @staticmethod
    def pot_bucket(pot_size: int) -> int:
        for i, bound in enumerate(POT_BUCKET_BOUNDS):
            if pot_size < bound:
                return i
        return POT_BUCKETS - 1

    @staticmethod
    def pot_odds(call_amount: int, pot: int) -> float:
        if call_amount == 0:
            return 0.0
        return call_amount / (pot + call_amount)
