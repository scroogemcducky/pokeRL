"""Play Kuhn poker in the terminal against an equilibrium bot.

A thin Textual UI over ``poker.kuhn``: every rule (legality, payoffs, what
each player may see) comes from ``KuhnState``; this module only displays it.

Usage::

    uv run python -m poker.play_kuhn

Examples
--------
>>> import random
>>> from poker.kuhn import Action, Card, KuhnState
>>> state = KuhnState(deal=(Card.KING, Card.JACK), history=(Action.BET_RAISE,))
>>> equilibrium_opponent(state, random.Random(0))  # J facing a bet folds
<Action.FOLD: 0>
"""

import argparse
import functools
import logging
import random
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widgets import Button, Footer, Static

from poker.kuhn import ALL_DEALS, Action, Card, Deal, KuhnState

logger = logging.getLogger(__name__)

type Opponent = Callable[[KuhnState], Action]

C, B = Action.CHECK_CALL, Action.BET_RAISE

# Probability of the aggressive option (bet, or call when facing a bet) for
# one Kuhn equilibrium (player 0 uses alpha = 0), keyed by (card, history).
_EQUILIBRIUM: dict[tuple[Card, tuple[Action, ...]], float] = {
    # player 0, first decision: always check
    (Card.JACK, ()): 0.0,
    (Card.QUEEN, ()): 0.0,
    (Card.KING, ()): 0.0,
    # player 1 after a check: bluff J 1/3, check Q, bet K
    (Card.JACK, (C,)): 1 / 3,
    (Card.QUEEN, (C,)): 0.0,
    (Card.KING, (C,)): 1.0,
    # player 1 facing a bet: fold J, call Q 1/3, call K
    (Card.JACK, (B,)): 0.0,
    (Card.QUEEN, (B,)): 1 / 3,
    (Card.KING, (B,)): 1.0,
    # player 0 facing a bet after checking: fold J, call Q 1/3, call K
    (Card.JACK, (C, B)): 0.0,
    (Card.QUEEN, (C, B)): 1 / 3,
    (Card.KING, (C, B)): 1.0,
}


def equilibrium_opponent(state: KuhnState, rng: random.Random) -> Action:
    """Choose an action for the current player from a Kuhn equilibrium.

    Sees only its own observation, never the other player's card.

    Parameters
    ----------
    state
        A non-terminal state.
    rng
        Source of randomness for the mixed decisions.
    """
    observation = state.observe(state.current_player)
    weak, strong = state.legal_actions()  # (check, bet) or (fold, call)
    p_strong = _EQUILIBRIUM[(observation.card, observation.history)]
    return strong if rng.random() < p_strong else weak


# Every color is "ansi_default", so the terminal's own theme shows through.
# (Theme's required ``dark`` flag only steers derived shades; with nothing but
# terminal defaults there is nothing to shade.)
_TERMINAL_THEME = Theme(
    name="terminal",
    ansi=True,
    primary="ansi_default",
    secondary="ansi_default",
    warning="ansi_default",
    error="ansi_default",
    success="ansi_default",
    accent="ansi_default",
    foreground="ansi_default",
    background="ansi_default",
    surface="ansi_default",
    panel="ansi_default",
    boost="ansi_default",
    variables={
        "ansi-background": "ansi_default",
        "ansi-foreground": "ansi_default",
    },
)


class KuhnApp(App[None]):
    """Heads-up Kuhn against ``opponent``; the human's seat alternates each hand."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("f", "act('fold')", "Fold"),
        ("c", "act('check_call')", "Check/Call"),
        ("b", "act('bet_raise')", "Bet"),
        ("n", "next_hand", "Next hand"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, deals: Iterator[Deal], opponent: Opponent) -> None:
        """Create the app.

        Parameters
        ----------
        deals
            Source of deals, one per hand.
        opponent
            Chooses the bot's action in a given state.
        """
        super().__init__(ansi_color=True)  # keep ANSI colors; no RGB conversion
        self.register_theme(_TERMINAL_THEME)
        self.theme = _TERMINAL_THEME.name
        self._deals = deals
        self._opponent = opponent
        self.human_seat = 0
        self.hands = 0
        self.score = 0
        self.state = KuhnState(deal=next(self._deals))

    def get_css_variables(self) -> dict[str, str]:
        """Replace every ANSI palette color Textual picks with the terminal default.

        Textual's ANSI mode still hard-codes a few palette entries (button
        borders, footer keys, scrollbars); this keeps all color choices with
        the terminal. Text attributes such as bold/dim/reverse are kept.
        """
        return {
            name: "ansi_default" if value.startswith("ansi_") else value
            for name, value in super().get_css_variables().items()
        }

    def compose(self) -> ComposeResult:
        """Build the widgets: status text, action buttons, key help."""
        yield Static(id="status")
        with Horizontal():
            for label, button_id in [
                ("Fold", "fold"),
                ("Check", "check_call"),
                ("Bet", "bet_raise"),
                ("Next hand", "next_hand"),
            ]:
                button = Button(label, id=button_id)
                button.can_focus = False  # play with keys or clicks; no focus marker
                yield button
        yield Footer()

    def on_mount(self) -> None:
        """Start the first hand."""
        self._start_hand()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button clicks to the same actions as the key bindings."""
        if event.button.id == "next_hand":
            self.action_next_hand()
        elif event.button.id is not None:
            self.action_act(event.button.id)

    def action_act(self, button_id: str) -> None:
        """Apply the human's action, then let the bot play until it's our turn."""
        if self.query_one(f"#{button_id}", Button).disabled:
            return
        self.state = self.state.step(Action[button_id.upper()])
        self._advance()

    def action_next_hand(self) -> None:
        """Deal the next hand, swapping seats, once the current one is over."""
        if not self.state.is_terminal():
            return
        self.human_seat = 1 - self.human_seat
        self.state = KuhnState(deal=next(self._deals))
        self._start_hand()

    def _start_hand(self) -> None:
        self.hands += 1
        self._advance()

    def _advance(self) -> None:
        """Play bot moves until the human acts or the hand ends; then redraw."""
        while (
            not self.state.is_terminal()
            and self.state.current_player != self.human_seat
        ):
            self.state = self.state.step(self._opponent(self.state))
        if self.state.is_terminal():
            self.score += self.state.returns()[self.human_seat]
            logger.debug("hand %d: %s", self.hands, self.state)
        self._redraw()

    def _redraw(self) -> None:
        legal = self.state.legal_actions()
        for action in Action:
            button = self.query_one(f"#{action.name.lower()}", Button)
            button.disabled = action not in legal
        self.query_one("#check_call", Button).label = (
            "Call" if Action.FOLD in legal else "Check"
        )
        self.query_one("#next_hand", Button).disabled = not self.state.is_terminal()
        self.query_one("#status", Static).update(self.status_text())

    def status_text(self) -> str:
        """Describe the hand from the human's point of view."""
        seen = self.state.observe(self.human_seat)
        position = "first" if self.human_seat == 0 else "second"
        history = " ".join(_describe(seen.history)) or "(no actions yet)"
        lines = [
            f"Hand {self.hands} - you act {position}",
            f"Your card:      {seen.card.name}",
            f"Opponent card:  {self._opponent_card()}",
            f"Actions:        {history}",
            f"Pot:            {sum(self.state.contributions())}",
        ]
        if self.state.is_terminal():
            won = self.state.returns()[self.human_seat]
            verdict = "You win" if won > 0 else "You lose"
            lines.append(f"{verdict} {abs(won)}.  Press n for the next hand.")
        lines.append(f"Score:          {self.score:+d} over {self.hands} hands")
        return "\n".join(lines)

    def _opponent_card(self) -> str:
        """Return the bot's card at showdown, else ``??`` (folds stay hidden)."""
        showdown = self.state.is_terminal() and self.state.history[-1] != Action.FOLD
        return self.state.deal[1 - self.human_seat].name if showdown else "??"


def _describe(history: Sequence[Action]) -> list[str]:
    """Name actions in poker terms: check vs call depends on a prior bet.

    Examples
    --------
    >>> _describe([Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL])
    ['check', 'bet', 'call']
    """
    words = []
    for i, action in enumerate(history):
        if action == Action.CHECK_CALL:
            words.append("call" if Action.BET_RAISE in history[:i] else "check")
        else:
            words.append("bet" if action == Action.BET_RAISE else "fold")
    return words


def _random_deals(rng: random.Random) -> Iterator[Deal]:
    while True:
        yield rng.choice(ALL_DEALS)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the terminal game.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="Play Kuhn poker vs a bot.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    rng = random.Random(args.seed)
    opponent = functools.partial(equilibrium_opponent, rng=rng)
    KuhnApp(deals=_random_deals(rng), opponent=opponent).run()


if __name__ == "__main__":
    main()
