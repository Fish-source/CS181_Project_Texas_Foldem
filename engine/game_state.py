from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from config import Stage, Action
from engine.card import Card


@dataclass
class PlayerState:
    hand: List[Card] = field(default_factory=list)
    chips: int = 1000
    current_bet: int = 0
    total_bet: int = 0
    is_folded: bool = False
    is_all_in: bool = False

    @property
    def is_active(self):
        return not self.is_folded and not self.is_all_in


@dataclass
class GameState:
    players: List[PlayerState] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)
    pot: int = 0
    stage: Stage = Stage.PREFLOP
    current_player: int = 0
    raises_this_street: int = 0
    dealer_idx: int = 0
    last_raiser: int = -1
    acted_this_street: List[bool] = field(default_factory=list)
    betting_history: List[Tuple[int, Action, int]] = field(default_factory=list)
    is_terminal: bool = False
    winner: Optional[int] = None

    @property
    def num_active_players(self):
        return sum(1 for p in self.players if not p.is_folded)

    def clone(self):
        import copy
        return copy.deepcopy(self)
