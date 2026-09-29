"""Acts as the correctness oracle for the Rust engine.

Every property of a state is derived from its deal and its public action history.

Examples
--------
>>> from poker.kuhn import ALL_DEALS
>>> len(ALL_DEALS)
6
"""

from dataclasses import dataclass
from enum import IntEnum
from itertools import permutations


class Card(IntEnum):
    """A Kuhn card. Integer order is showdown strength: J < Q < K."""

    JACK = 0
    QUEEN = 1
    KING = 2


class Action(IntEnum):
    """A Kuhn action."""

    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


type Deal = tuple[Card, Card]
"""``(player 0's card, player 1's card)``."""

ALL_DEALS: tuple[Deal, ...] = tuple(permutations(Card, 2))
"""Every possible deal: the six ordered pairs of distinct cards."""


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


def _is_terminal(history: tuple[Action, ...]) -> bool:
    """Return whether the betting round is over.

    It ends on a fold, or on a check/call that is not the opening action
    (check-check, bet-call, check-bet-call).

    Examples
    --------
    >>> C, B = Action.CHECK_CALL, Action.BET_RAISE
    >>> _is_terminal((C,)), _is_terminal((C, C)), _is_terminal((C, B))
    (False, True, False)
    """
    if not history:
        return False
    last = history[-1]
    return last == Action.FOLD or (last == Action.CHECK_CALL and len(history) >= 2)


def _legal_actions(history: tuple[Action, ...]) -> tuple[Action, ...]:
    """Return the legal actions after ``history``, in ``Action`` order.

    Facing the single allowed bet: fold or call. Otherwise: check or bet.

    Examples
    --------
    >>> _legal_actions(())
    (<Action.CHECK_CALL: 1>, <Action.BET_RAISE: 2>)
    >>> _legal_actions((Action.BET_RAISE,))
    (<Action.FOLD: 0>, <Action.CHECK_CALL: 1>)
    """
    if _is_terminal(history):
        return ()
    if Action.BET_RAISE in history:
        return (Action.FOLD, Action.CHECK_CALL)
    return (Action.CHECK_CALL, Action.BET_RAISE)


@dataclass(frozen=True)
class KuhnObservation:
    """What the player can see: their own card and the public history."""

    player: int
    card: Card
    history: tuple[Action, ...]


@dataclass(frozen=True)
class KuhnState:
    """A(n immutable) Kuhn hand: a deal plus the actions taken so far.

    Examples
    --------
    >>> state = KuhnState(deal=(Card.JACK, Card.KING))
    >>> state.history
    ()
    """

    deal: Deal
    history: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        """Reject impossible deals and histories that break the rules."""
        if self.deal[0] == self.deal[1]:
            msg = f"deal must hold two distinct cards, got {self.deal}"
            raise ValueError(msg)
        for i, action in enumerate(self.history):
            if action not in _legal_actions(self.history[:i]):
                msg = f"{action!r} is illegal after {self.history[:i]}"
                raise IllegalActionError(msg)

    def is_terminal(self) -> bool:
        """Return whether the hand is over."""
        return _is_terminal(self.history)

    def legal_actions(self) -> tuple[Action, ...]:
        """Return the actions the current player may take (empty if terminal)."""
        return _legal_actions(self.history)

    @property
    def current_player(self) -> int:
        """The player to act: 0 or 1. Players alternate, starting with 0.

        Raises
        ------
        ValueError
            If the hand is terminal (nobody acts).

        Examples
        --------
        >>> KuhnState(deal=(Card.JACK, Card.KING)).current_player
        0
        """
        if self.is_terminal():
            msg = "terminal state has no current player"
            raise ValueError(msg)
        return len(self.history) % 2

    def step(self, action: Action) -> "KuhnState":
        """Return the state after the current player takes ``action``.

        The original state is unchanged.

        Raises
        ------
        IllegalActionError
            If ``action`` is not in ``legal_actions()``, including any action
            after the hand has ended.
        """
        if action not in self.legal_actions():
            msg = f"{action!r} is illegal after {self.history}"
            raise IllegalActionError(msg)
        return KuhnState(deal=self.deal, history=(*self.history, action))

    def contributions(self) -> tuple[int, int]:
        """Return the chips each player has put in so far: ante, bet, call.

        Examples
        --------
        >>> state = KuhnState(deal=(Card.JACK, Card.KING), history=(Action.BET_RAISE,))
        >>> state.contributions()
        (2, 1)
        """
        chips = [1, 1]  # antes
        for i, action in enumerate(self.history):
            is_call = (
                action == Action.CHECK_CALL and Action.BET_RAISE in self.history[:i]
            )
            if action == Action.BET_RAISE or is_call:
                chips[i % 2] += 1
        return (chips[0], chips[1])

    def returns(self) -> tuple[int, int]:
        """Return each player's net chip gain for the finished hand.

        The winner takes what the loser put in, so the returns sum to zero.

        Raises
        ------
        ValueError
            If the hand is not terminal.

        Examples
        --------
        >>> C = Action.CHECK_CALL
        >>> KuhnState(deal=(Card.JACK, Card.KING), history=(C, C)).returns()
        (-1, 1)
        """
        if not self.is_terminal():
            msg = f"hand is not terminal after {self.history}"
            raise ValueError(msg)
        if self.history[-1] == Action.FOLD:
            loser = (len(self.history) - 1) % 2  # whoever folded
        else:
            loser = 0 if self.deal[0] < self.deal[1] else 1  # showdown
        won = self.contributions()[loser]
        return (-won, won) if loser == 0 else (won, -won)

    def observe(self, player: int) -> KuhnObservation:
        """Return what ``player`` can see: their own card and the public history.

        Built only from visible fields (an allowlist), so the opponent's card
        cannot leak in.

        Raises
        ------
        ValueError
            If ``player`` is not 0 or 1.

        Examples
        --------
        >>> state = KuhnState(deal=(Card.JACK, Card.KING))
        >>> state.observe(1).card
        <Card.KING: 2>
        """
        if player not in (0, 1):
            msg = f"player must be 0 or 1, got {player}"
            raise ValueError(msg)
        return KuhnObservation(
            player=player, card=self.deal[player], history=self.history
        )
