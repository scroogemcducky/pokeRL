"""Poker: poker reinforcement learning with a Rust simulator.

Examples
--------
>>> import poker
>>> poker.__version__ == poker._native.version()
True
"""

from importlib.metadata import version as _distribution_version

from poker import _native

__version__: str = _distribution_version("poker")

__all__ = ["__version__", "_native"]
