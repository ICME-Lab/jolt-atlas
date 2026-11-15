//! The field items the external `akita-*` crates bind as `jolt_field::*`, re-exported from
//! [`atlas_algebra::field`]. Nothing in the atlas crates names this package; it exists while
//! `akita-*` is an external dependency.

pub use atlas_algebra::field::{
    canonical_extension_basis, is_registered_prime_offset, pseudo_mersenne_modulus,
    registered_prime_offset_spec, solinas, AdditiveGroup, CanonicalBytes, CanonicalEncoding, Ext2,
    Ext2Config, ExtField, Field, Fold, Fp128, Fp128x8i32, Fp32, Fp64, FpExt2, FpExt4, FpExt8,
    MulBaseUnreduced, One, Packed, Prime128Offset275, Prime128OffsetA7F7, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59, PrimeOffsetSpec, PseudoMersenne, Ring, Unreduced,
    WithCommitAccumulator, Zero, PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX,
    PRIME_OFFSET_SPECS,
};
pub use atlas_algebra::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};

#[cfg(test)]
mod tests {
    /// `akita-*` bounds its field parameters on this package's traits; atlas's field must
    /// satisfy them as the type `akita-*` is instantiated with.
    fn akita_bound<F: crate::Field + crate::CanonicalEncoding + crate::PseudoMersenne>() {}

    #[test]
    fn atlas_fp128_satisfies_the_akita_bound() {
        akita_bound::<atlas_algebra::field::fp128::Fp128>();
    }
}
