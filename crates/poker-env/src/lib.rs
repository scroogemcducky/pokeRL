//! Authoritative poker simulator.
//!
//! This crate owns all poker semantics (cards, betting, legal actions,
//! payoffs). It has no Python dependency so it can be tested with plain
//! `cargo test`.

/// Returns the crate version, taken from `Cargo.toml` at compile time.
///
/// Used by the Python package to check that the compiled extension matches
/// the source it was built from.
///
/// # Examples
///
/// ```
/// assert_eq!(poker_env::version(), env!("CARGO_PKG_VERSION"));
/// ```
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_package_version() {
        assert_eq!(version(), "0.1.0");
    }
}
