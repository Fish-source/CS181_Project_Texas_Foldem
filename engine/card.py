import random
from enum import IntEnum

from treys import Card as TreysCard


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


_SUIT_SYMBOLS = {Suit.CLUBS: 'c', Suit.DIAMONDS: 'd', Suit.HEARTS: 'h', Suit.SPADES: 's'}
_RANK_SYMBOLS = {
    Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
    Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
    Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K', Rank.ACE: 'A',
}


class Card:
    __slots__ = ('suit', 'rank', 'treys_card')

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
        self.treys_card = TreysCard.new(_RANK_SYMBOLS[rank] + _SUIT_SYMBOLS[suit])

    def __repr__(self):
        return _RANK_SYMBOLS[self.rank] + _SUIT_SYMBOLS[self.suit]

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))


ALL_CARDS = [Card(s, r) for s in Suit for r in Rank]


class Deck:
    def __init__(self):
        self._cards = []
        self.reset()

    def reset(self):
        self._cards = ALL_CARDS[:]
        random.shuffle(self._cards)

    def deal(self, n: int = 1):
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    @property
    def remaining(self):
        return len(self._cards)
