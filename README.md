# RL environment for Poker

Poker reinforcement learning: a Rust poker simulator,
Python orchestration, and MLX for Apple Silicon. Architecture loosely inspired by [Pufferlib](https://github.com/PufferAI/PufferLib) (added as a subtree for agents to reference).

## Layout

```text
crates/poker-env/  pure Rust simulator
crates/poker-py/   thin PyO3 bindings -> poker._native
python/poker/     Python package (orchestration, training, evaluation)
tests/            pytest suite
```

## Setup

Requires Rust and uv.

```bash
uv sync                 # creates .venv, builds the Rust extension
```

## Tests

```bash
cargo test
uv run pytest
```
