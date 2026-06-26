//! Prefix components of lookup table MLE decompositions.
//!
//! Prefixes capture the high-order bits of lookup table inputs and maintain checkpoints
//! that accumulate contributions during the prefix-suffix sum-check protocol. By evaluating
//! prefix MLEs incrementally with checkpoints updated every two rounds, we avoid summing
//! over the full 2^(2*XLEN) lookup table.

use self::{
    and::AndPrefix, eq::EqPrefix, less_than::LessThanPrefix,
    lower_word_no_msb::LowerWordNoMsbPrefix, not_msb::NotMsbPrefix, or::OrPrefix,
    word_lt_bound::WordLtBoundPrefix, xor::XorPrefix, zero_gt_bound::ZeroGtBoundPrefix,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{
        clamp::{CLAMP_OP1_LOWER, CLAMP_OP2_LOWER, CLAMP_OPS_UPPER},
        prefixes::{msb::MsbPrefix, nlw::NotLowerWordPrefix},
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
/// Lower word (without MSB) prefix implementation.
pub mod lower_word_no_msb;
/// MSB (most significant bit) prefix implementation.
pub mod msb;
/// Two's complement negation prefix: `(!lower_word) + 1`, used in `neg_relu` decomposition.
pub mod nlw;
/// Not-MSB (most significant bit) prefix implementation.
pub mod not_msb;
/// Bitwise OR prefix implementation.
pub mod or;
/// Prefix that checks all bits with significance >= given bound are less than the bound.
pub mod word_lt_bound;
/// Bitwise XOR prefix implementation.
pub mod xor;
/// Prefix that checks all bits with significance >= given bound are zero.
pub mod zero_gt_bound;

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
        #[derive(EnumCountMacro, EnumIter, FromPrimitive, Copy, Clone, Debug)]
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

// Type aliases for specific prefix implementations with configured parameters.
type OpsWordLtHighPrefix<const XLEN: usize> =
    WordLtBoundPrefix<XLEN, CLAMP_OPS_UPPER, { Prefixes::OpsWordLtHigh as usize }>;
type Op1WordLtLowPrefix<const XLEN: usize> =
    WordLtBoundPrefix<XLEN, CLAMP_OP1_LOWER, { Prefixes::Op1WordLtLow as usize }>;
type Op2WordLtLowPrefix<const XLEN: usize> =
    WordLtBoundPrefix<XLEN, CLAMP_OP2_LOWER, { Prefixes::Op2WordLtLow as usize }>;
type OpsZeroGtHighPrefix<const XLEN: usize> =
    ZeroGtBoundPrefix<XLEN, CLAMP_OPS_UPPER, { Prefixes::OpsZeroGtHigh as usize }>;
type Op1ZeroGtLowPrefix<const XLEN: usize> =
    ZeroGtBoundPrefix<XLEN, CLAMP_OP1_LOWER, { Prefixes::Op1ZeroGtLow as usize }>;
type Op2ZeroGtLowPrefix<const XLEN: usize> =
    ZeroGtBoundPrefix<XLEN, CLAMP_OP2_LOWER, { Prefixes::Op2ZeroGtLow as usize }>;

impl_sparse_dense_prefix!(
    And             : AndPrefix,                // Bitwise AND prefix
    Eq              : EqPrefix,                 // Equality comparison prefix
    LessThan        : LessThanPrefix,           // Less-than comparison prefix
    LowerWordNoMsb  : LowerWordNoMsbPrefix,     // Lower word without MSB prefix
    NotMsb          : NotMsbPrefix,             // Not-MSB prefix
    Or              : OrPrefix,                 // Bitwise OR prefix
    Xor             : XorPrefix,                // Bitwise XOR prefix
    Msb             : MsbPrefix,                // MSB (sign bit) prefix
    NotLowerWord    : NotLowerWordPrefix,       // Two's complement negation prefix: `(!lower_word) + 1`
    OpsWordLtHigh   : OpsWordLtHighPrefix,      // ∑_{i < HighBound} x_i * 2^i
    OpsZeroGtHigh   : OpsZeroGtHighPrefix,      // ∀i >= HighBound, x_i == 0
    Op1WordLtLow    : Op1WordLtLowPrefix,       // ∑_{i < Op1LowBound} x_i * 2^i
    Op2WordLtLow    : Op2WordLtLowPrefix,       // ∑_{i < Op2LowBound} x_i * 2^i
    Op1ZeroGtLow    : Op1ZeroGtLowPrefix,       // ∀i >= Op1LowBound, x_i == 0
    Op2ZeroGtLow    : Op2ZeroGtLowPrefix,       // ∀i >= Op2LowBound, x_i == 0
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

impl<F: JoltField> PrefixEval<F> {
    pub fn inner(&self) -> F {
        self.0
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
