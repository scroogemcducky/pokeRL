"""Generate the Kuhn golden-trajectory file from the oracle.

Every complete hand (all deals x all legal action paths) is written as a JSON
record the Rust engine can be tested against. Enum values are written by
name, so the file is readable and does not depend on enum numbering.

Usage::

    uv run python -m poker.kuhn_fixtures --out tests/fixtures/kuhn/trajectories.json

Examples
--------
>>> len(all_trajectories())
30
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from poker.kuhn import ALL_DEALS, KuhnState

logger = logging.getLogger(__name__)


class ObservationRecord(TypedDict):
    """JSON form of a ``KuhnObservation``."""

    player: int
    card: str
    history: list[str]


class StepRecord(TypedDict):
    """One decision: who acts, what is legal, and what the actor sees."""

    actor: int
    legal_actions: list[str]
    observation: ObservationRecord


class TrajectoryRecord(TypedDict):
    """One complete hand, step by step."""

    deal: list[str]
    actions: list[str]
    steps: list[StepRecord]
    returns: list[int]


def _terminal_states(state: KuhnState) -> list[KuhnState]:
    """Return every terminal state reachable from ``state``, depth-first."""
    if state.is_terminal():
        return [state]
    return [t for a in state.legal_actions() for t in _terminal_states(state.step(a))]


def trajectory(terminal: KuhnState) -> TrajectoryRecord:
    """Replay a finished hand and record every decision along the way.

    Examples
    --------
    >>> from poker.kuhn import Action, Card
    >>> hand = KuhnState(deal=(Card.KING, Card.JACK), history=(Action.BET_RAISE,))
    >>> trajectory(hand.step(Action.FOLD))["returns"]
    [1, -1]
    """
    steps: list[StepRecord] = []
    state = KuhnState(deal=terminal.deal)
    for action in terminal.history:
        actor = state.current_player
        observation = state.observe(actor)
        steps.append(
            {
                "actor": actor,
                "legal_actions": [a.name for a in state.legal_actions()],
                "observation": {
                    "player": observation.player,
                    "card": observation.card.name,
                    "history": [a.name for a in observation.history],
                },
            }
        )
        state = state.step(action)
    return {
        "deal": [card.name for card in terminal.deal],
        "actions": [a.name for a in terminal.history],
        "steps": steps,
        "returns": list(terminal.returns()),
    }


def all_trajectories() -> list[TrajectoryRecord]:
    """Return every complete Kuhn hand in a fixed order (deal, then depth-first)."""
    return [
        trajectory(terminal)
        for deal in ALL_DEALS
        for terminal in _terminal_states(KuhnState(deal=deal))
    ]


def main(argv: Sequence[str] | None = None) -> None:
    """Write the golden-trajectory file.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    records = all_trajectories()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    logger.info("wrote %d trajectories to %s", len(records), args.out)


if __name__ == "__main__":
    main()
