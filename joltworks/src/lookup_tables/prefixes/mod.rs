//! Prefix components of lookup table MLE decompositions.
//!
//! Prefixes capture the high-order bits of lookup table inputs and maintain checkpoints
//! that accumulate contributions during the prefix-suffix sum-check protocol. By evaluating
//! prefix MLEs incrementally with checkpoints updated every two rounds, we avoid summing
//! over the full 2^(2*XLEN) lookup table.

use self::{and::AndPrefix, eq::EqPrefix, less_than::LessThanPrefix, or::OrPrefix, xor::XorPrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{
        lower_msb::LowerMsbPrefix, lower_word_no_msb::LowerWordNoMsbPrefix, msb::MsbPrefix,
        not_lower_msb::NotLowerMsbPrefix, not_msb::NotMsbPrefix,
        not_word_no_msb::NotWordNoMsbPrefix, sat_val::SatValPrefix, upper_eqo::UpperEqoPrefix,
        upper_eqz::UpperEqzPrefix, word_no_msb::WordNoMsbPrefix,
    },
    utils::lookup_bits::LookupBits,
};
use common::parallel::par_enabled;
use num::FromPrimitive;
use num_derive::FromPrimitive;
use rayon::prelude::*;
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

/// Bitwise AND prefix implementation.
pub mod and;
/// Equality comparison prefix implementation.
pub mod eq;
/// Less-than comparison prefix implementation.
pub mod less_than;
/// Lower 32-bit word sign bit (i32 sign bit) prefix implementation.
pub mod lower_msb;
/// Lower word without MSB prefix implementation (64-bit layout, sign at position 32).
pub mod lower_word_no_msb;
/// MSB (most significant bit) prefix implementation.
pub mod msb;
/// Complement of the lower 32-bit word sign bit prefix implementation.
pub mod not_lower_msb;
/// Not-MSB (most significant bit) prefix implementation.
pub mod not_msb;
/// Two's complement negation prefix: `(!lower_word) + 1`, used in `neg_relu` decomposition.
pub mod not_word_no_msb;
/// Bitwise OR prefix implementation.
pub mod or;
/// Saturated boundary value prefix implementation, used in `sat_clamp` decomposition.
pub mod sat_val;
/// Upper half-word all-ones (eqo) prefix implementation.
pub mod upper_eqo;
/// Upper half-word all-zeros (eqz) prefix implementation.
pub mod upper_eqz;
/// Lower word (without MSB) prefix implementation.
pub mod word_no_msb;
/// Bitwise XOR prefix implementation.
pub mod xor;

/// Trait for prefix components that support sparse-dense MLE evaluation.
///
/// Prefixes are evaluated during the sum-check protocol using checkpoints that
/// accumulate contributions from previously bound variables. This enables efficient
/// prover implementation without iterating over the full lookup table.
pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    /// Evalautes the MLE for this prefix:
    /// - prefix(r, r_x, c, b)   if j is odd
    /// - prefix(r, c, b)        if j is even
    ///
    /// where the prefix checkpoint captures the "contribution" of
    /// `r` to this evaluation.
    ///
    /// `r` (and potentially `r_x`) capture the variables of the prefix
    /// that have been bound in the previous rounds of sumcheck.
    /// To compute the current round's prover message, we're fixing the
    /// current variable to `c`.
    /// The remaining variables of the prefix are captured by `b`. We sum
    /// over these variables as they range over the Boolean hypercube, so
    /// they can be represented by a single bitvector.
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
}

macro_rules! impl_sparse_dense_prefix {
    ($($name:ident: $prefix:ident),* $(,)?) => {

        /// An enum containing all prefixes used by Jolt's instruction lookup tables.
        #[repr(u8)]
        #[derive(EnumCountMacro, EnumIter, FromPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Prefixes {
            $($name,)*
        }

       impl Prefixes {
            /// Evalautes the MLE for this prefix:
            /// - prefix(r, r_x, c, b)   if j is odd
            /// - prefix(r, c, b)        if j is even
            ///
            /// where the prefix checkpoint captures the "contribution" of
            /// `r` to this evaluation.
            ///
            /// `r` (and potentially `r_x`) capture the variables of the prefix
            /// that have been bound in the previous rounds of sumcheck.
            /// To compute the current round's prover message, we're fixing the
            /// current variable to `c`.
            /// The remaining variables of the prefix are captured by `b`. We sum
            /// over these variables as they range over the Boolean hypercube, so
            /// they can be represented by a single bitvector.
            pub fn prefix_mle<const XLEN: usize, F, C>(
                &self,
                checkpoints: &PrefixCheckpoints<F>,
                r_x: Option<C>,
                c: u32,
                b: LookupBits,
                j: usize,
            ) -> PrefixEval<F>
            where
                C: ChallengeFieldOps<F>,
                F: JoltField + FieldChallengeOps<C>,
            {
                let eval = match self {
                    $(Prefixes::$name => $prefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),)*
                };
                PrefixEval(eval)
            }

            /// Every two rounds of sumcheck, we update the "checkpoint" value for each
            /// prefix, incorporating the two random challenges `r_x` and `r_y` received
            /// since the last update.
            /// This function updates all the prefix checkpoints.
            #[tracing::instrument(skip_all)]
            pub fn update_checkpoints<const XLEN: usize, F, C>(
                checkpoints: &mut PrefixCheckpoints<F>,
                r_x: C,
                r_y: C,
                j: usize,
                suffix_len: usize,
            ) where
                C: ChallengeFieldOps<F>,
                F: JoltField + FieldChallengeOps<C>,
            {
                debug_assert_eq!(checkpoints.len(), Self::COUNT);
                let previous_checkpoints = checkpoints.clone();
                checkpoints
                    .0
                    .par_iter_mut()
                    .with_min_len(par_enabled())
                    .enumerate()
                    .for_each(|(index, new_checkpoint)| {
                        let prefix: Self = FromPrimitive::from_u8(index as u8).unwrap();
                        *new_checkpoint = prefix.update_prefix_checkpoint::<XLEN, F, C>(
                            &previous_checkpoints,
                            r_x,
                            r_y,
                            j,
                            suffix_len,
                        );
                    });
            }

            /// Every two rounds of sumcheck, we update the "checkpoint" value for each
            /// prefix, incorporating the two random challenges `r_x` and `r_y` received
            /// since the last update.
            /// `j` is the sumcheck round index.
            /// A checkpoint update may depend on the values of the other prefix checkpoints,
            /// so we pass in all such `checkpoints` to this function.
            fn update_prefix_checkpoint<const XLEN: usize, F, C>(
                &self,
                checkpoints: &PrefixCheckpoints<F>,
                r_x: C,
                r_y: C,
                j: usize,
                suffix_len: usize,
            ) -> PrefixCheckpoint<F>
            where
                C: ChallengeFieldOps<F>,
                F: JoltField + FieldChallengeOps<C>,
            {
                match self {
                    $(Prefixes::$name => $prefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len),)*

                }
            }
        }
    };
}

impl_sparse_dense_prefix!(
    And             : AndPrefix,                // Bitwise AND prefix
    Eq              : EqPrefix,                 // Equality comparison prefix
    LessThan        : LessThanPrefix,           // Less-than comparison prefix
    WordNoMsb       : WordNoMsbPrefix,     // Lower word without MSB prefix
    NotMsb          : NotMsbPrefix,             // Not-MSB prefix
    Or              : OrPrefix,                 // Bitwise OR prefix
    Xor             : XorPrefix,                // Bitwise XOR prefix
    Msb             : MsbPrefix,                // MSB (sign bit) prefix
    NotWordNoMsb    : NotWordNoMsbPrefix,       // Two's complement negation prefix: `(!lower_word) + 1`
    SatVal          : SatValPrefix,             // Saturated boundary value prefix: `m*MIN + (1-m)*MAX`, used in `sat_clamp` decomposition
    UpperEqz        : UpperEqzPrefix,           // Upper half-word all-zeros (eqz) prefix, used in `sat_clamp` decomposition
    UpperEqo        : UpperEqoPrefix,           // Upper half-word all-ones (eqo) prefix, used in `sat_clamp` decomposition
    NotLowerMsb     : NotLowerMsbPrefix,        // Complement of the lower 32-bit word sign bit (1 − r[XLEN/2]), used in `sat_clamp` decomposition
    LowerMsb        : LowerMsbPrefix,           // Lower 32-bit word sign bit (r[XLEN/2], the i32 sign bit), used in `sat_clamp` decomposition
    LowerWordNoMsb  : LowerWordNoMsbPrefix,     // Lower word without MSB prefix (64-bit layout), used in `sat_clamp` decomposition
);

#[derive(Clone, Copy)]
/// Wrapper for prefix polynomial evaluations, used for type safety in prefix operations.
pub struct PrefixEval<F>(F);
/// Optional prefix evaluation cached after each pair of address-binding rounds (r_x, r_y).
pub type PrefixCheckpoint<F: JoltField> = PrefixEval<Option<F>>;

#[derive(Clone)]
// Stores the checkpoints for all prefixes, updated every two rounds of sumcheck.
pub struct PrefixCheckpoints<F: JoltField>([PrefixCheckpoint<F>; Prefixes::COUNT]);

impl<F: JoltField> PrefixCheckpoints<F> {
    pub fn new() -> Self {
        Self(std::array::from_fn(|_| None.into()))
    }

    pub fn len(&self) -> usize {
        Prefixes::COUNT
    }
}

impl<F: JoltField> Default for PrefixCheckpoints<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField> std::ops::Mul<F> for PrefixEval<F> {
    type Output = F;

    fn mul(self, rhs: F) -> Self::Output {
        self.0 * rhs
    }
}
impl<F: JoltField> std::ops::Mul<PrefixEval<F>> for PrefixEval<F> {
    type Output = F;

    fn mul(self, rhs: PrefixEval<F>) -> Self::Output {
        self.0 * rhs.0
    }
}

impl<F: Display> Display for PrefixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> From<F> for PrefixEval<F> {
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F> PrefixCheckpoint<F> {
    /// Unwraps the optional checkpoint value, panicking if None.
    pub fn unwrap(self) -> PrefixEval<F> {
        self.0.unwrap().into()
    }
}

impl<F: JoltField> Index<usize> for PrefixCheckpoints<F> {
    type Output = Option<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index].0
    }
}

impl<F: JoltField> IndexMut<usize> for PrefixCheckpoints<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index].0
    }
}

impl<F: JoltField> Index<Prefixes> for PrefixCheckpoints<F> {
    type Output = Option<F>;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        &self[prefix as usize]
    }
}

impl<F: JoltField> IndexMut<Prefixes> for PrefixCheckpoints<F> {
    fn index_mut(&mut self, prefix: Prefixes) -> &mut Self::Output {
        &mut self[prefix as usize]
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}
