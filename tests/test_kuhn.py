"""Tests for the Python Kuhn poker oracle, ``poker.kuhn``.

The expected values in these tests are derived by hand from the rules of
Kuhn poker.

Rules (Kuhn, 1950): three cards J < Q < K; each player antes 1 chip and gets
one private card; one betting round with a fixed 1-chip bet. Player 0 acts
first. There are exactly five terminal histories::

    c c      check, check          showdown for the 1-chip antes
    b c      bet, call             showdown for 2 chips
    b f      bet, fold             player 0 wins the ante
    c b c    check, bet, call      showdown for 2 chips
    c b f    check, bet, fold      player 1 wins the ante
"""

from collections.abc import Callable, Sequence
from itertools import permutations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from poker.kuhn import (
    ALL_DEALS,
    Action,
    Card,
    Deal,
    IllegalActionError,
    KuhnObservation,
    KuhnState,
)

type Play = Callable[[Deal, Sequence[Action]], KuhnState]

J, Q, K = Card.JACK, Card.QUEEN, Card.KING
F, C, B = Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE

TERMINAL_HISTORIES = [(C, C), (B, C), (B, F), (C, B, C), (C, B, F)]


# --- 1. Deals -------------------------------------------------------------


def test_all_deals_are_the_six_ordered_pairs_of_distinct_cards() -> None:
    assert sorted(ALL_DEALS) == sorted(permutations([J, Q, K], 2))
    assert len(ALL_DEALS) == 6


def test_deal_with_duplicate_card_is_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        KuhnState(deal=(Q, Q))


# --- 2. Terminal payoffs (hand-derived) -----------------------------------


@pytest.mark.parametrize(
    ("deal", "history", "expected"),
    [
        # check-check: showdown for the antes, higher card wins 1
        ((K, J), (C, C), (1, -1)),
        ((J, K), (C, C), (-1, 1)),
        # bet-call: showdown for ante + bet, higher card wins 2
        ((Q, J), (B, C), (2, -2)),
        ((J, K), (B, C), (-2, 2)),
        # bet-fold: bettor (P0) wins the ante whatever the cards
        ((J, K), (B, F), (1, -1)),
        # check-bet-call: showdown for 2
        ((K, Q), (C, B, C), (2, -2)),
        ((Q, K), (C, B, C), (-2, 2)),
        # check-bet-fold: bettor (P1) wins the ante whatever the cards
        ((K, J), (C, B, F), (-1, 1)),
    ],
)
def test_terminal_payoffs(
    play: Play,
    deal: tuple[Card, Card],
    history: tuple[Action, ...],
    expected: tuple[int, int],
) -> None:
    state = play(deal, history)
    assert state.is_terminal()
    assert state.returns() == expected


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ((), (1, 1)),
        ((C,), (1, 1)),
        ((B,), (2, 1)),
        ((C, B), (1, 2)),
        ((C, C), (1, 1)),
        ((B, C), (2, 2)),
        ((B, F), (2, 1)),
        ((C, B, C), (2, 2)),
        ((C, B, F), (1, 2)),
    ],
)
def test_contributions_after_each_history(
    play: Play, history: tuple[Action, ...], expected: tuple[int, int]
) -> None:
    assert play((J, Q), history).contributions() == expected


# --- 3. Actor and legal actions after every history -----------------------


@pytest.mark.parametrize(
    ("history", "actor", "legal"),
    [
        ((), 0, (C, B)),  # opening: check or bet; nothing to fold to
        ((C,), 1, (C, B)),  # after a check: check or bet
        ((B,), 1, (F, C)),  # facing a bet: fold or call; no raise in Kuhn
        ((C, B), 0, (F, C)),  # facing a bet after checking: fold or call
    ],
)
def test_actor_and_legal_actions(
    play: Play,
    history: tuple[Action, ...],
    actor: int,
    legal: tuple[Action, ...],
) -> None:
    state = play((J, Q), history)
    assert not state.is_terminal()
    assert state.current_player == actor
    assert state.legal_actions() == legal


@pytest.mark.parametrize("history", TERMINAL_HISTORIES)
def test_terminal_state_has_no_actor_and_no_legal_actions(
    play: Play, history: tuple[Action, ...]
) -> None:
    state = play((J, Q), history)
    assert state.legal_actions() == ()
    with pytest.raises(ValueError, match="terminal"):
        _ = state.current_player


# --- 4. Rejecting invalid use ---------------------------------------------


@pytest.mark.parametrize(
    ("history", "illegal"),
    [
        ((), F),  # cannot fold when nothing is owed
        ((C,), F),
        ((B,), B),  # no raise in Kuhn
        ((C, B), B),
    ],
)
def test_illegal_action_is_rejected(
    play: Play, history: tuple[Action, ...], illegal: Action
) -> None:
    state = play((J, Q), history)
    with pytest.raises(IllegalActionError):
        state.step(illegal)


def test_step_after_terminal_is_rejected(play: Play) -> None:
    with pytest.raises(IllegalActionError):
        play((J, Q), (C, C)).step(C)


def test_returns_before_terminal_are_rejected(play: Play) -> None:
    with pytest.raises(ValueError, match="not terminal"):
        play((J, Q), (B,)).returns()


def test_constructing_state_with_illegal_history_is_rejected() -> None:
    with pytest.raises(IllegalActionError):
        KuhnState(deal=(J, Q), history=(F,))


# --- 5. Immutability and determinism --------------------------------------


def test_step_returns_new_state_and_leaves_original_unchanged() -> None:
    start = KuhnState(deal=(J, Q))
    after = start.step(B)
    assert start.history == ()
    assert after.history == (B,)
    assert after is not start


def test_same_deal_and_actions_give_equal_states(play: Play) -> None:
    assert play((K, Q), (C, B, C)) == play((K, Q), (C, B, C))


# --- 6. Observations and hidden information -------------------------------


def test_observation_contains_own_card_and_public_history(play: Play) -> None:
    state = play((J, K), (C, B))
    assert state.observe(0) == KuhnObservation(player=0, card=J, history=(C, B))
    assert state.observe(1) == KuhnObservation(player=1, card=K, history=(C, B))


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("history", [(), (C,), (B,), (C, B), *TERMINAL_HISTORIES])
def test_observation_does_not_depend_on_opponent_card(
    play: Play, player: int, history: tuple[Action, ...]
) -> None:
    """Changing only the opponent's card must not change what a player sees.

    For each own card there are two possible opponent cards; both deals must
    produce identical observations for ``player``.
    """
    for own in (J, Q, K):
        deals = [d for d in ALL_DEALS if d[player] == own]
        observations = {play(d, history).observe(player) for d in deals}
        assert len(observations) == 1


@pytest.mark.parametrize("player", [-1, 2])
def test_observe_rejects_invalid_player(player: int) -> None:
    with pytest.raises(ValueError, match="player"):
        KuhnState(deal=(J, Q)).observe(player)


# --- 7. Whole-game enumeration --------------------------------------------


def test_there_are_exactly_thirty_complete_hands(
    all_terminal_states: list[KuhnState],
) -> None:
    assert len(all_terminal_states) == 30
    histories = {s.history for s in all_terminal_states}
    assert histories == set(TERMINAL_HISTORIES)


def test_every_complete_hand_is_zero_sum_and_conserves_chips(
    all_terminal_states: list[KuhnState],
) -> None:
    for state in all_terminal_states:
        r0, r1 = state.returns()
        c0, c1 = state.contributions()
        assert r0 + r1 == 0
        # The winner gains exactly what the loser put in.
        assert (r0, r1) in {(c1, -c1), (-c0, c0)}


# --- 8. Properties over random play (hypothesis) --------------------------


# Useless for this small game (have full enumeration of all states), just trying.
@given(deal=st.sampled_from(ALL_DEALS), data=st.data())
def test_random_legal_play_ends_within_three_actions_and_is_zero_sum(
    deal: tuple[Card, Card], data: st.DataObject
) -> None:
    state = KuhnState(deal=deal)
    steps = 0
    while not state.is_terminal():
        legal = state.legal_actions()
        assert legal, "a non-terminal state must have a legal action"
        state = state.step(data.draw(st.sampled_from(legal)))
        steps += 1
        assert steps <= 3
    assert sum(state.returns()) == 0
    assert 2 <= steps <= 3
