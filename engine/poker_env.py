from typing import Dict, List, Tuple, Optional

from config import (
    Stage, Action, NUM_PLAYERS, INITIAL_CHIPS,
    SMALL_BLIND, BIG_BLIND, COMMUNITY_CARDS_BY_STAGE,
)
from engine.card import Card, Deck
from engine.game_state import GameState, PlayerState
from engine.rule_engine import RuleEngine


class PokerEnv:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.state: Optional[GameState] = None
        self.deck: Optional[Deck] = None
        self._hand_number = 0

    def reset(self) -> Dict:
        self._hand_number += 1
        self.deck = Deck()
        self.deck.reset()

        self.state = GameState(
            players=[PlayerState(chips=INITIAL_CHIPS) for _ in range(NUM_PLAYERS)],
            community_cards=[],
            pot=0,
            stage=Stage.PREFLOP,
            current_player=0,
            raises_this_street=0,
            dealer_idx=(self._hand_number - 1) % NUM_PLAYERS,
            last_raiser=-1,
            acted_this_street=[False] * NUM_PLAYERS,
            betting_history=[],
            is_terminal=False,
            winner=None,
        )

        for i in range(NUM_PLAYERS):
            self.state.players[i].hand = self.deck.deal(2)

        sb_idx = (self.state.dealer_idx + 1) % NUM_PLAYERS
        bb_idx = (self.state.dealer_idx + 2) % NUM_PLAYERS

        sb_amount = min(SMALL_BLIND, self.state.players[sb_idx].chips)
        self.state.players[sb_idx].chips -= sb_amount
        self.state.players[sb_idx].current_bet = sb_amount
        self.state.players[sb_idx].total_bet = sb_amount
        self.state.pot += sb_amount

        bb_amount = min(BIG_BLIND, self.state.players[bb_idx].chips)
        self.state.players[bb_idx].chips -= bb_amount
        self.state.players[bb_idx].current_bet = bb_amount
        self.state.players[bb_idx].total_bet = bb_amount
        self.state.pot += bb_amount

        self.state.current_player = (bb_idx + 1) % NUM_PLAYERS if NUM_PLAYERS > 2 else sb_idx

        if NUM_PLAYERS == 2:
            self.state.current_player = sb_idx

        return self.observe(self.state.current_player)

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        if self.state.is_terminal:
            return self.observe(self.state.current_player), 0.0, True, {}

        player_idx = self.state.current_player
        legal = self.rule_engine.get_legal_actions(self.state)

        if action not in legal:
            action = legal[0] if Action.FOLD in legal else legal[0]

        old_stage = self.state.stage
        self.state = self.rule_engine.apply_action(self.state, action)
        new_stage = self.state.stage

        if not self.state.is_terminal and new_stage != old_stage:
            self.deal_community_cards()

        if self.state.is_terminal:
            payoffs = self.rule_engine.compute_payoffs(self.state)
            obs = self.observe(player_idx)
            reward = float(payoffs[player_idx]) / INITIAL_CHIPS
            return obs, reward, True, {'payoffs': payoffs, 'winner': self.state.winner}

        obs = self.observe(self.state.current_player)
        return obs, 0.0, False, {}

    def observe(self, player_idx: int) -> Dict:
        if self.state is None:
            return {}

        player = self.state.players[player_idx]
        opp_idx = 1 - player_idx
        opp = self.state.players[opp_idx]

        obs = {
            'hand': list(player.hand),
            'community_cards': list(self.state.community_cards),
            'pot': self.state.pot,
            'stage': self.state.stage,
            'my_bet': player.current_bet,
            'opp_bet': opp.current_bet,
            'my_chips': player.chips,
            'opp_chips': opp.chips,
            'my_total_bet': player.total_bet,
            'opp_total_bet': opp.total_bet,
            'is_folded': player.is_folded,
            'opp_folded': opp.is_folded,
            'legal_actions': self.rule_engine.get_legal_actions(self.state),
            'current_player': self.state.current_player,
            'dealer_idx': self.state.dealer_idx,
            'raises_this_street': self.state.raises_this_street,
            'betting_history': list(self.state.betting_history),
        }
        return obs

    @property
    def current_player(self) -> int:
        if self.state is None:
            return 0
        return self.state.current_player

    @property
    def is_terminal(self) -> bool:
        if self.state is None:
            return False
        return self.state.is_terminal

    def deal_community_cards(self):
        if self.state is None or self.state.is_terminal:
            return

        current_count = len(self.state.community_cards)
        target_count = COMMUNITY_CARDS_BY_STAGE.get(self.state.stage, current_count)
        needed = target_count - current_count

        if needed > 0 and self.deck is not None:
            new_cards = self.deck.deal(needed)
            self.state.community_cards.extend(new_cards)
