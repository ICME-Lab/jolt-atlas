//! Prefix components of lookup table MLE decompositions.
//!
//! Prefixes capture the high-order bits of lookup table inputs and maintain checkpoints
//! that accumulate contributions during the prefix-suffix sum-check protocol. By evaluating
//! prefix MLEs incrementally with checkpoints updated every two rounds, we avoid summing
//! over the full 2^(2*XLEN) lookup table.

use crate::onnx_proof::lookup_tables::prefixes::{
    and::AndPrefix, eq::EqPrefix, less_than::LessThanPrefix,
    lower_word_no_msb::LowerWordNoMsbPrefix, not_msb::NotMsbPrefix, or::OrPrefix, xor::XorPrefix,
};
use joltworks::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};
use num::FromPrimitive;
use num_derive::FromPrimitive;
use rayon::prelude::*;
use std::{fmt::Display, ops::Index};
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
/// Not-MSB (most significant bit) prefix implementation.
pub mod not_msb;
/// Bitwise OR prefix implementation.
pub mod or;
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
        checkpoints: &[PrefixCheckpoint<F>],
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
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
}

/// An enum containing all prefixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Prefixes {
    /// Bitwise AND prefix
    And,
    /// Equality comparison prefix
    Eq,
    /// Less-than comparison prefix
    LessThan,
    /// Lower word without MSB prefix
    LowerWordNoMsb,
    /// Not-MSB prefix
    NotMsb,
    /// Bitwise OR prefix
    Or,
    /// Bitwise XOR prefix
    Xor,
}

#[derive(Clone, Copy)]
/// Wrapper for prefix polynomial evaluations, used for type safety in prefix operations.
pub struct PrefixEval<F>(F);
/// Optional prefix evaluation cached after each pair of address-binding rounds (r_x, r_y).
pub type PrefixCheckpoint<F: JoltField> = PrefixEval<Option<F>>;

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

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
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
        checkpoints: &[PrefixCheckpoint<F>],
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
            Prefixes::And => AndPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Eq => EqPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LessThan => LessThanPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LowerWordNoMsb => {
                LowerWordNoMsbPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::NotMsb => NotMsbPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Or => OrPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Xor => XorPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
        };
        PrefixEval(eval)
    }

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// This function updates all the prefix checkpoints.
    #[tracing::instrument(skip_all)]
    pub fn update_checkpoints<const XLEN: usize, F, C>(
        checkpoints: &mut [PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(checkpoints.len(), Self::COUNT);
        let previous_checkpoints = checkpoints.to_vec();
        checkpoints
            .par_iter_mut()
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
        checkpoints: &[PrefixCheckpoint<F>],
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
            Prefixes::And => {
                AndPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Eq => {
                EqPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LessThan => {
                LessThanPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LowerWordNoMsb => LowerWordNoMsbPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::NotMsb => {
                NotMsbPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Or => {
                OrPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Xor => {
                XorPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
        }
    }
}
