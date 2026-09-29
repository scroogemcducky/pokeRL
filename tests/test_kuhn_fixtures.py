"""Tests for the Kuhn golden-trajectory file, ``tests/fixtures/kuhn/``.

The file is generated from the oracle, so it does not prove the oracle
correct (the hand-derived tests in ``test_kuhn.py`` do that). It is the
language-neutral contract the Rust engine is checked against, and a change
detector: any change in oracle behaviour fails here and shows up as a diff.

Regenerate with::

    uv run python -m poker.kuhn_fixtures --out tests/fixtures/kuhn/trajectories.json
"""

from pathlib import Path
from typing import Any

from poker.kuhn_fixtures import all_trajectories, main


def test_golden_file_matches_oracle(kuhn_golden: list[Any]) -> None:
    assert kuhn_golden == all_trajectories()


def test_golden_file_has_every_complete_hand_once(kuhn_golden: list[Any]) -> None:
    hands = {(tuple(t["deal"]), tuple(t["actions"])) for t in kuhn_golden}
    assert len(kuhn_golden) == 30
    assert len(hands) == 30


def test_one_trajectory_record_matches_hand_derivation(
    kuhn_golden: list[Any],
) -> None:
    """Pin the record format with one hand written out in full by hand.

    Deal J/K, check-bet-call: showdown for 2, K wins.
    """
    expected = {
        "deal": ["JACK", "KING"],
        "actions": ["CHECK_CALL", "BET_RAISE", "CHECK_CALL"],
        "steps": [
            {
                "actor": 0,
                "legal_actions": ["CHECK_CALL", "BET_RAISE"],
                "observation": {"player": 0, "card": "JACK", "history": []},
            },
            {
                "actor": 1,
                "legal_actions": ["CHECK_CALL", "BET_RAISE"],
                "observation": {
                    "player": 1,
                    "card": "KING",
                    "history": ["CHECK_CALL"],
                },
            },
            {
                "actor": 0,
                "legal_actions": ["FOLD", "CHECK_CALL"],
                "observation": {
                    "player": 0,
                    "card": "JACK",
                    "history": ["CHECK_CALL", "BET_RAISE"],
                },
            },
        ],
        "returns": [-2, 2],
    }
    assert expected in kuhn_golden


def test_cli_writes_the_committed_file(tmp_path: Path, golden_kuhn_path: Path) -> None:
    out = tmp_path / "trajectories.json"
    main(["--out", str(out)])
    assert out.read_text() == golden_kuhn_path.read_text()
