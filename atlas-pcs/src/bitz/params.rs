//! The shape and soundness parameters of a BitZ opening.
//!
//! Soundness is round-by-round: every challenge is preceded by proof of work that brings its
//! round's error to `2^-target_bits`. The terms are the projection onto a random prime
//! (below), the sumcheck and batching rounds over `F_{2^128}` (a few bits each, set where
//! they are drawn) and Ligerito's queries (no proof of work: the query count is chosen for the
//! target).

use super::ligerito::Schedule;
use super::pack::LOG_PACKING;

/// The fewest and most variables of a committed bit vector.
pub const MIN_VARS: usize = 12;
pub const MAX_VARS: usize = 31;

/// The bits of Fp128's modulus.
const FIELD_BITS: u32 = 128;
/// The primes are drawn from `[2^(high − PRIME_INTERVAL_BITS), 2^high)`.
const PRIME_INTERVAL_BITS: u32 = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    /// `log2` of the number of committed bits.
    pub num_vars: usize,
    /// The tensor split: `2^log_rows` rows by `2^log_columns` columns.
    pub log_rows: usize,
    pub log_columns: usize,
    pub target_bits: u32,
    /// The projection prime is uniform among the primes in `[2^low, 2^high)`.
    pub prime_low_bits: u32,
    pub prime_high_bits: u32,
    /// The proof of work before the prime is drawn.
    pub projection_pow_bits: u32,
    pub ligerito: Schedule,
}

impl Params {
    /// The parameters for `2^num_vars` bits at `target_bits`, with rows the square root of the
    /// bit count (the split the paper recommends at 128 bits: the prime may be larger).
    pub fn new(num_vars: usize, target_bits: u32) -> Option<Self> {
        let ligerito = Schedule::new(num_vars.checked_sub(LOG_PACKING)?, target_bits)?;
        Self::with_schedule(num_vars, target_bits, ligerito)
    }

    /// As [`Self::new`] with a given Ligerito schedule; `None` if the schedule does not reach
    /// the target.
    pub fn with_schedule(num_vars: usize, target_bits: u32, ligerito: Schedule) -> Option<Self> {
        if !(MIN_VARS..=MAX_VARS).contains(&num_vars)
            || ligerito.num_vars + LOG_PACKING != num_vars
            || ligerito.target_bits != target_bits
            || ligerito.query_error_bits() < target_bits as f64
        {
            return None;
        }
        let log_rows = num_vars.div_ceil(2).max(LOG_PACKING);
        // The grand product proves integer equalities in the exponent of a generator of order
        // 2^128 − 1: a sum of 2^log_rows residues must stay below it.
        let prime_high_bits = FIELD_BITS - log_rows as u32;
        let prime_low_bits = prime_high_bits - PRIME_INTERVAL_BITS;
        let mut params = Self {
            num_vars,
            log_rows,
            log_columns: num_vars - log_rows,
            target_bits,
            prime_low_bits,
            prime_high_bits,
            projection_pow_bits: 0,
            ligerito,
        };
        let deficit = target_bits as f64 - params.projection_error_bits();
        params.projection_pow_bits = deficit.ceil().max(0.0) as u32;
        Some(params)
    }

    /// `−log2` of the projection's error (the paper's `κ_0`, Theorem 4.7 with Lemma 4.3 for a
    /// prime field): `log2(M) / (|P| · ⌊log2 q_min⌋)`, where a false lifted claim differs from
    /// the true one by a nonzero integer of absolute value at most `M = 2·ℓ·(p − 1)²`, which
    /// at most `log2(M) / ⌊log2 q_min⌋` of the primes divide.
    pub fn projection_error_bits(&self) -> f64 {
        let log_bound = 1.0 + self.num_vars as f64 + 2.0 * FIELD_BITS as f64;
        // π(x) > x / ln x for x ≥ 17 and π(x) < 1.25506·x / ln x for x > 1 (Rosser and
        // Schoenfeld), so the interval holds at least this many primes, as a multiple of 2^high.
        let (low, high) = (self.prime_low_bits as f64, self.prime_high_bits as f64);
        let ln2 = std::f64::consts::LN_2;
        let density = 1.0 / (high * ln2) - 1.25506 * (low - high).exp2() / (low * ln2);
        let log_primes = high + density.log2();
        log_primes + low.log2() - log_bound.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_cover_the_supported_sizes() {
        for num_vars in MIN_VARS..=MAX_VARS {
            let params = Params::new(num_vars, 128).unwrap();
            assert_eq!(params.log_rows + params.log_columns, num_vars);
            assert!(params.log_rows >= LOG_PACKING);
            // A column sum of residues stays below the generator's order.
            assert!(params.prime_high_bits as usize + params.log_rows <= 128);
            assert!(params.prime_high_bits <= 127);
        }
        assert!(Params::new(MIN_VARS - 1, 128).is_none());
        assert!(Params::new(MAX_VARS + 1, 128).is_none());
    }

    #[test]
    fn the_projection_is_ground_to_the_target() {
        // At 2^30 bits: rows 2^15, primes in [2^108, 2^113), error about 2^-105 (the paper's
        // Table 20 gives 2^-105.3 at its optimal interval), 23 bits of proof of work.
        let params = Params::new(30, 128).unwrap();
        assert_eq!(
            (
                params.log_rows,
                params.prime_low_bits,
                params.prime_high_bits
            ),
            (15, 108, 113)
        );
        let bits = params.projection_error_bits();
        assert!((104.9..105.5).contains(&bits), "{bits}");
        assert_eq!(params.projection_pow_bits, 23);
        // At a 100-bit target nothing is ground.
        assert_eq!(Params::new(30, 100).unwrap().projection_pow_bits, 0);
    }

    #[test]
    fn a_schedule_below_the_target_is_rejected() {
        let mut schedule = Schedule::new(23, 128).unwrap();
        // 314 queries at rate 1/2 give (3/4)^314 = 2^-130.3, 2^-128.3 over the four levels.
        assert_eq!((schedule.levels(), schedule.queries), (4, 314));
        assert!(schedule.query_error_bits() >= 128.0);
        assert!(Params::with_schedule(30, 128, schedule.clone()).is_some());
        // 300 queries give (3/4)^300 = 2^-124.5, 2^-122.5 over the four levels.
        schedule.queries = 300;
        let bits = schedule.query_error_bits();
        assert!((122.4..122.6).contains(&bits), "{bits}");
        assert!(Params::with_schedule(30, 128, schedule).is_none());
    }
}
