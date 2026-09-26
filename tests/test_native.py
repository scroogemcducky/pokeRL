"""Tests for the compiled Rust extension, ``poker._native``."""

import poker
from poker import _native


def test_native_version_matches_package_version() -> None:
    """The compiled extension must come from the same source as the package.

    A mismatch means the Rust extension is stale: Rust code changed but
    ``maturin develop`` was not rerun.
    """
    assert _native.version() == poker.__version__
