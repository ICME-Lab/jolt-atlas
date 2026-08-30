//! Parallel [`DoryRoutines`] for the prover.
//!
//! The arkworks backend's vector routines (`fixed_base_vector_scalar_mul`,
//! `fixed_scalar_mul_bases_then_add`, `fixed_scalar_mul_vs_then_add`) are
//! sequential loops of one scalar multiplication per element; in the
//! reduce-and-fold rounds they scale vectors of `2^σ` G1 *and* G2 points, so
//! they dominate the evaluation proof on a multi-core prover. These wrappers
//! run them element-wise in parallel and delegate the MSM to the backend.
use common::parallel::par_enabled;
use dory::{
    backends::arkworks::{ArkFr, ArkG1, ArkG2, G1Routines, G2Routines},
    primitives::arithmetic::{DoryRoutines, Group},
};
use rayon::prelude::*;

/// [`G1Routines`] with parallel vector operations.
pub struct ParG1Routines;

/// [`G2Routines`] with parallel vector operations.
pub struct ParG2Routines;

macro_rules! par_routines {
    ($name:ident, $group:ty, $inner:ty) => {
        impl DoryRoutines<$group> for $name {
            fn msm(bases: &[$group], scalars: &[ArkFr]) -> $group {
                <$inner as DoryRoutines<$group>>::msm(bases, scalars)
            }

            fn fixed_base_vector_scalar_mul(base: &$group, scalars: &[ArkFr]) -> Vec<$group> {
                scalars
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|s| base.scale(s))
                    .collect()
            }

            fn fixed_scalar_mul_bases_then_add(
                bases: &[$group],
                vs: &mut [$group],
                scalar: &ArkFr,
            ) {
                assert_eq!(bases.len(), vs.len(), "Lengths must match");
                vs.par_iter_mut()
                    .zip(bases.par_iter())
                    .with_min_len(par_enabled())
                    .for_each(|(v, base)| *v = v.add(&base.scale(scalar)));
            }

            fn fixed_scalar_mul_vs_then_add(vs: &mut [$group], addends: &[$group], scalar: &ArkFr) {
                assert_eq!(vs.len(), addends.len(), "Lengths must match");
                vs.par_iter_mut()
                    .zip(addends.par_iter())
                    .with_min_len(par_enabled())
                    .for_each(|(v, addend)| *v = v.scale(scalar).add(addend));
            }
        }
    };
}

par_routines!(ParG1Routines, ArkG1, G1Routines);
par_routines!(ParG2Routines, ArkG2, G2Routines);
