use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{PrefixCheckpoints, Prefixes},
    utils::lookup_bits::LookupBits,
};
use common::consts::TRIG_PERIOD_MODULUS;

/// `bit` if `k_bit` is set, else `1 - bit` — the per-position term of `EQ(x, K)`.
fn eq_term<F: JoltField>(bit: F, k_bit: bool) -> F {
    if k_bit {
        bit
    } else {
        F::one() - bit
    }
}

/// Running product `Π_{j<=i} [x_j if k_j=1 else (1-x_j)]`: are the bound bits of `x` equal to
/// `K`'s corresponding bits so far. Self-contained (no cross-prefix dependency), unlike
/// [`LessThanConstPrefix`] below.
pub enum EqConstPrefix<const XLEN: usize, const K: u64, const CP_INDEX: usize> {}

impl<const XLEN: usize, const K: u64, const CP_INDEX: usize> EqConstPrefix<XLEN, K, CP_INDEX> {
    /// The `i`-th bit of `K` (MSB-first), truncated to `XLEN` bits. `false` for `i >= XLEN`,
    /// since every registered prefix's checkpoint updates every round regardless of which
    /// table is being proven.
    fn k_bit(i: usize) -> bool {
        i < XLEN && (K >> (XLEN - 1 - i)) & 1 == 1
    }
}

impl<const XLEN: usize, const K: u64, const CP_INDEX: usize, F: JoltField> SparseDensePrefix<F>
    for EqConstPrefix<XLEN, K, CP_INDEX>
{
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let r_x: F = r_x.into();
            if j > 0 {
                result *= eq_term(r_x, Self::k_bit(j - 1));
            }
        }
        result *= eq_term(F::from_u32(c), Self::k_bit(j));

        let b_len = b.len();
        let b_u64: u64 = b.into();
        for pos in 0..b_len {
            let global_index = j + 1 + pos;
            let bit = F::from_u8(((b_u64 >> (b_len - 1 - pos)) & 1) as u8);
            result *= eq_term(bit, Self::k_bit(global_index));
        }

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());
        if j > 0 {
            result *= eq_term(r_x.into(), Self::k_bit(j - 1));
        }
        result *= eq_term(r_y.into(), Self::k_bit(j));
        Some(result).into()
    }
}

/// Running sum `Σ_{i: k_i=1} (1-x_i) · Π_{j<i} EQ(x_j,k_j)`: the "less than constant `K`"
/// comparator, reading [`EqConstPrefix`]'s *previous* checkpoint at `EQ_CP_INDEX` the same way
/// `LessThanPrefix` reads `Prefixes::Eq`.
pub enum LessThanConstPrefix<
    const XLEN: usize,
    const K: u64,
    const EQ_CP_INDEX: usize,
    const LT_CP_INDEX: usize,
> {}

impl<const XLEN: usize, const K: u64, const EQ_CP_INDEX: usize, const LT_CP_INDEX: usize>
    LessThanConstPrefix<XLEN, K, EQ_CP_INDEX, LT_CP_INDEX>
{
    /// See [`EqConstPrefix::k_bit`] — identical formula, duplicated per-type so callers don't
    /// need to spell out `XLEN`/`K` at each call site.
    fn k_bit(i: usize) -> bool {
        i < XLEN && (K >> (XLEN - 1 - i)) & 1 == 1
    }
}

impl<
        const XLEN: usize,
        const K: u64,
        const EQ_CP_INDEX: usize,
        const LT_CP_INDEX: usize,
        F: JoltField,
    > SparseDensePrefix<F> for LessThanConstPrefix<XLEN, K, EQ_CP_INDEX, LT_CP_INDEX>
{
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut lt = checkpoints[LT_CP_INDEX].unwrap_or(F::zero());
        let mut eq = checkpoints[EQ_CP_INDEX].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let r_x: F = r_x.into();
            if j > 0 {
                let idx = j - 1;
                if Self::k_bit(idx) {
                    lt += eq * (F::one() - r_x);
                }
                eq *= eq_term(r_x, Self::k_bit(idx));
            }
            let c = F::from_u32(c);
            if Self::k_bit(j) {
                lt += eq * (F::one() - c);
            }
            eq *= eq_term(c, Self::k_bit(j));
        } else {
            let c = F::from_u32(c);
            if Self::k_bit(j) {
                lt += eq * (F::one() - c);
            }
            eq *= eq_term(c, Self::k_bit(j));
        }

        let b_len = b.len();
        let b_u64: u64 = b.into();
        for pos in 0..b_len {
            let global_index = j + 1 + pos;
            let bit = F::from_u8(((b_u64 >> (b_len - 1 - pos)) & 1) as u8);
            if Self::k_bit(global_index) {
                lt += eq * (F::one() - bit);
            }
            eq *= eq_term(bit, Self::k_bit(global_index));
        }

        lt
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut lt = checkpoints[LT_CP_INDEX].unwrap_or(F::zero());
        let mut eq = checkpoints[EQ_CP_INDEX].unwrap_or(F::one());

        let r_x: F = r_x.into();
        let r_y: F = r_y.into();
        if j > 0 {
            let idx = j - 1;
            if Self::k_bit(idx) {
                lt += eq * (F::one() - r_x);
            }
            eq *= eq_term(r_x, Self::k_bit(idx));
        }
        if Self::k_bit(j) {
            lt += eq * (F::one() - r_y);
        }

        Some(lt).into()
    }
}

/// `EqConstPrefix` specialized to `K = TRIG_PERIOD_MODULUS`, for the trig remainder
/// range-check's single-operand `LessThanConstTable`.
pub type TrigEqConstPrefix<const XLEN: usize> =
    EqConstPrefix<XLEN, { TRIG_PERIOD_MODULUS as u64 }, { Prefixes::TrigEqConst as usize }>;

/// `LessThanConstPrefix` specialized to `K = TRIG_PERIOD_MODULUS`.
pub type TrigLessThanConstPrefix<const XLEN: usize> = LessThanConstPrefix<
    XLEN,
    { TRIG_PERIOD_MODULUS as u64 },
    { Prefixes::TrigEqConst as usize },
    { Prefixes::TrigLessThanConst as usize },
>;
