//! PyO3 bindings exposing `poker-env` to Python as `poker._native`.
//!
//! This crate is deliberately thin: it converts between Python and Rust
//! types and forwards calls. Poker logic never lives here.

use pyo3::prelude::*;

#[pymodule]
mod _native {
    use pyo3::prelude::*;

    /// Return the version of the compiled Rust simulator.
    #[pyfunction]
    fn version() -> &'static str {
        poker_env::version()
    }
}
