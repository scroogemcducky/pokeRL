"""Shared pytest fixtures.

Fixtures here are available to every test module without importing.
"""

from collections.abc import Callable, Sequence

import pytest
from poker.kuhn import ALL_DEALS, Action, Deal, KuhnState

type Play = Callable[[Deal, Sequence[Action]], KuhnState]


def _play(deal: Deal, actions: Sequence[Action]) -> KuhnState:
    """Start a hand with ``deal`` and apply ``actions`` in order."""
    state = KuhnState(deal=deal)
    for action in actions:
        state = state.step(action)
    return state


def _walk(state: KuhnState) -> list[KuhnState]:
    """Return every terminal state reachable from ``state`` (depth-first)."""
    if state.is_terminal():
        return [state]
    return [
        terminal
        for action in state.legal_actions()
        for terminal in _walk(state.step(action))
    ]


@pytest.fixture
def play() -> Play:
    """Return a helper that plays a list of actions from a fresh deal."""
    return _play


@pytest.fixture(scope="session")
def all_terminal_states() -> list[KuhnState]:
    """Every complete Kuhn hand: all deals x all legal action paths."""
    return [terminal for deal in ALL_DEALS for terminal in _walk(KuhnState(deal=deal))]
