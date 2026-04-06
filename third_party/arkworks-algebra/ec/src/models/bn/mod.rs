use crate::{
    models::{short_weierstrass::SWCurveConfig, CurveConfig},
    pairing::{CompressedPairing, MillerLoopOutput, Pairing, PairingOutput},
};
use ark_ff::{
    fields::{
        fp12_2over3over2::{Fp12, Fp12Config},
        fp2::Fp2Config,
        fp6_3over2::Fp6Config,
        Field, Fp2, PrimeField,
    },
    CyclotomicMultSubgroup,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_chunks_mut, marker::PhantomData, vec::*};
use educe::Educe;
use itertools::Itertools;
use num_traits::One;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// For debugging purposes.
// const FACTOR1: [u64; 3] = [0x3a422f764fffffff, 0x8df6ed4bc8923d8a, 0x3bec47df15e307c8];
// const FACTOR2: [u64; 3] = [0xb830738ad2b0eed6, 0x1ea96b02d9d9e38d, 0x3bec47df15e307c8];
// const FACTOR3: [u64; 3] = [0x420398f3678302b8, 0x1ea96b02d9d9e38e, 0x3bec47df15e307c8];
// const FACTOR4: [u64; 3] = [0xb830738ad2b0eed5, 0x1ea96b02d9d9e38d, 0x3bec47df15e307c8];

pub enum TwistType {
    M,
    D,
}

/// Optimized version of raise_to_sixth_cyclotomic_polynomial where the input is already in a cyclotomic subgroup.
fn pow_sixth_cyclotomic_polynomial_over_r_cyclotomic_optimized<P: BnConfig>(
    mut f: Fp12<P::Fp12Config>,
) -> Fp12<P::Fp12Config> {
    // https://cacr.uwaterloo.ca/techreports/2011/cacr2011-26.pdf
    // Hard part follows Laura Fuentes-Castaneda et al. "Faster hashing to G2"
    // by computing:
    //
    // result = elt^(q^3 * (12*z^3 + 6z^2 + 4z - 1) +
    //               q^2 * (12*z^3 + 6z^2 + 6z) +
    //               q   * (12*z^3 + 6z^2 + 4z) +
    //               1   * (12*z^3 + 12z^2 + 6z + 1))
    // which equals
    //
    // result = elt^( 2z * ( 6z^2 + 3z + 1 ) * (q^4 - q^2 + 1)/r ).

    let y0 = Bn::<P>::exp_by_neg_x(f);
    // y1 = f^{-2x}
    let y1 = y0.cyclotomic_square();
    let y2 = y1.cyclotomic_square();
    let mut y3 = y2 * &y1;

    // y4 = f^{6x^2}
    let y4 = Bn::<P>::exp_by_neg_x(y3);
    let y5 = y4.cyclotomic_square();
    let mut y6 = Bn::<P>::exp_by_neg_x(y5);

    // y3 = f^{6x}
    y3.cyclotomic_inverse_in_place();

    // y6 = f^{12x^3}
    y6.cyclotomic_inverse_in_place();

    let y7 = y6 * &y4;
    // y8 = a
    let mut y8 = y7 * &y3;

    // y9 = b = a * f^{-2x}
    let y9 = y8 * &y1;

    let y10 = y8 * &y4;
    // y11 is the first factor for the equation at the bottom of p.6
    let y11 = y10 * &f;

    // y12 is b^{p}
    let mut y12 = y9;
    y12.frobenius_map_in_place(1);

    // y13 is product of first two factors for the equation at the bottom of p.6
    let y13 = y12 * &y11;

    // y8 = a^p^2
    y8.frobenius_map_in_place(2);

    // y14 is product of first three factors for the equation at the bottom of p.6
    let y14 = y8 * &y13;
    f.cyclotomic_inverse_in_place();

    // y15 = b * f^{-1}
    let mut y15 = f * &y9;
    y15.frobenius_map_in_place(3);
    let y16 = y15 * &y14;

    y16
}

pub fn pow_sixth_cyclotomic_polynomial_over_r<P: BnConfig>(
    mut f: Fp12<P::Fp12Config>,
) -> Fp12<P::Fp12Config> {
    // https://cacr.uwaterloo.ca/techreports/2011/cacr2011-26.pdf
    // Hard part follows Laura Fuentes-Castaneda et al. "Faster hashing to G2"
    // by computing:
    //
    // result = elt^(q^3 * (12*z^3 + 6z^2 + 4z - 1) +
    //               q^2 * (12*z^3 + 6z^2 + 6z) +
    //               q   * (12*z^3 + 6z^2 + 4z) +
    //               1   * (12*z^3 + 12z^2 + 6z + 1))
    // which equals
    //
    // result = elt^( 2z * ( 6z^2 + 3z + 1 ) * (q^4 - q^2 + 1)/r ).

    // y0 = f^-x
    let y0 = Bn::<P>::exp_by_neg_x_non_cyclotomic(f);
    // y1 = f^{-2x}
    let y1 = y0.square();
    // y2 = f^{-4x}
    let y2 = y1.square();
    // y3 = f^{-6x}
    let mut y3 = y2 * &y1;

    // y4 = f^{6x^2}
    let y4 = Bn::<P>::exp_by_neg_x_non_cyclotomic(y3);
    // y5 = f^{12x^2}
    let y5 = y4.square();
    // y6 = f^{-12x^3}
    let mut y6 = Bn::<P>::exp_by_neg_x_non_cyclotomic(y5);

    // y3 = f^{6x}
    y3.inverse_in_place();

    // y6 = f^{12x^3}
    y6.inverse_in_place();

    let y7 = y6 * &y4;
    // y8 = a
    let mut y8 = y7 * &y3;

    // y9 = b = a * f^{-2x}
    let y9 = y8 * &y1;

    let y10 = y8 * &y4;

    // y11 is the first factor for the equation at the bottom of p.6
    let y11 = y10 * &f;

    // y12 is b^{p}
    let mut y12 = y9;
    y12.frobenius_map_in_place(1);

    // y13 is product of first two factors for the equation at the bottom of p.6
    let y13 = y12 * &y11;

    // y8 = a^p^2
    y8.frobenius_map_in_place(2);

    // y14 is product of first three factors for the equation at the bottom of p.6
    let y14 = y8 * &y13;
    f.inverse_in_place();

    // y15 = b * f^{-1}
    let mut y15 = f * &y9;
    y15.frobenius_map_in_place(3);
    let y16 = y15 * &y14;

    y16
}

pub fn raise_to_psi_six_pow<P: BnConfig>(f: Fp12<P::Fp12Config>) -> Option<Fp12<P::Fp12Config>> {
    // Beuchat et al p.9 equation 3.
    // https://eprint.iacr.org/2010/354.pdf

    // result = elt^((q^6-1)*(q^2+1)).
    // Follows, e.g., Beuchat et al page 9, by computing result as follows:
    //   elt^((q^6-1)*(q^2+1)) = (conj(elt) * elt^(-1))^(q^2+1)

    // f1 = r.cyclotomic_inverse_in_place() = f^(p^6)
    let mut f1 = f;
    f1.cyclotomic_inverse_in_place();

    f.inverse().map(|mut f2| {
        // f2 = f^(-1);
        // r = f^(p^6 - 1)
        let mut r = f1 * &f2;

        // f2 = f^(p^6 - 1)
        f2 = r;
        // r = f^((p^6 - 1)(p^2))
        r.frobenius_map_in_place(2);

        // r = f^((p^6 - 1)(p^2) + (p^6 - 1))
        // r = f^((p^6 - 1)(p^2 + 1))
        r * &f2
    })
}

pub trait FromPsi6Pow<P: BnConfig>: Sized {
    /// Compresses the psi^6 power of a Fp12 element to a CompressedFp12 element.
    /// The default implementation returns None, which does not enable the compressed pairing feature.
    fn from_psi_six_pow(_base: Fp12<P::Fp12Config>) -> Option<Self> {
        None
    }
}

pub trait BnConfig: 'static + Sized {
    /// The absolute value of the BN curve parameter `X`
    /// (as in `q = 36 X^4 + 36 X^3 + 24 X^2 + 6 X + 1`).
    const X: &'static [u64];

    /// Whether or not `X` is negative.
    const X_IS_NEGATIVE: bool;

    /// The absolute value of `6X + 2`.
    const ATE_LOOP_COUNT: &'static [i8];

    const TWIST_TYPE: TwistType;
    const TWIST_MUL_BY_Q_X: Fp2<Self::Fp2Config>;
    const TWIST_MUL_BY_Q_Y: Fp2<Self::Fp2Config>;
    type Fp: PrimeField + Into<<Self::Fp as PrimeField>::BigInt>;
    type Fp2Config: Fp2Config<Fp = Self::Fp>;
    type Fp6Config: Fp6Config<Fp2Config = Self::Fp2Config>;
    type Fp12Config: Fp12Config<Fp6Config = Self::Fp6Config>;
    type CompressedFp12Config: FromPsi6Pow<Self>
        + Sized
        + Sync
        + CanonicalDeserialize
        + CanonicalSerialize;
    type G1Config: SWCurveConfig<BaseField = Self::Fp>;
    type G2Config: SWCurveConfig<
        BaseField = Fp2<Self::Fp2Config>,
        ScalarField = <Self::G1Config as CurveConfig>::ScalarField,
    >;

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<G1Prepared<Self>>>,
        b: impl IntoIterator<Item = impl Into<G2Prepared<Self>>>,
    ) -> MillerLoopOutput<Bn<Self>> {
        let mut pairs = a
            .into_iter()
            .zip_eq(b)
            .filter_map(|(p, q)| {
                let (p, q) = (p.into(), q.into());
                match !p.is_zero() && !q.is_zero() {
                    true => Some((p, q.ell_coeffs.into_iter())),
                    false => None,
                }
            })
            .collect::<Vec<_>>();

        let mut f = cfg_chunks_mut!(pairs, 4)
            .map(|pairs| {
                let mut f = <Bn<Self> as Pairing>::TargetField::one();
                for i in (1..Self::ATE_LOOP_COUNT.len()).rev() {
                    if i != Self::ATE_LOOP_COUNT.len() - 1 {
                        f.square_in_place();
                    }

                    for (p, coeffs) in pairs.iter_mut() {
                        Bn::<Self>::ell(&mut f, &coeffs.next().unwrap(), &p.0);
                    }

                    let bit = Self::ATE_LOOP_COUNT[i - 1];
                    if bit == 1 || bit == -1 {
                        for (p, coeffs) in pairs.iter_mut() {
                            Bn::<Self>::ell(&mut f, &coeffs.next().unwrap(), &p.0);
                        }
                    }
                }
                f
            })
            .product::<<Bn<Self> as Pairing>::TargetField>();

        if Self::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }

        for (p, coeffs) in &mut pairs {
            Bn::<Self>::ell(&mut f, &coeffs.next().unwrap(), &p.0);
        }

        for (p, coeffs) in &mut pairs {
            Bn::<Self>::ell(&mut f, &coeffs.next().unwrap(), &p.0);
        }

        MillerLoopOutput(f)
    }

    #[allow(clippy::let_and_return)]
    fn final_exponentiation(f: MillerLoopOutput<Bn<Self>>) -> Option<PairingOutput<Bn<Self>>> {
        raise_to_psi_six_pow::<Self>(f.0)
            .map(pow_sixth_cyclotomic_polynomial_over_r_cyclotomic_optimized::<Self>)
            .map(PairingOutput)
    }

    fn compressed_final_exponentiation(
        f: MillerLoopOutput<Bn<Self>>,
    ) -> Option<Self::CompressedFp12Config> {
        let val = pow_sixth_cyclotomic_polynomial_over_r::<Self>(f.0);
        Self::CompressedFp12Config::from_psi_six_pow(val)
    }
}

pub mod g1;
pub mod g2;

pub use self::{
    g1::{G1Affine, G1Prepared, G1Projective},
    g2::{G2Affine, G2Prepared, G2Projective},
};

#[derive(Educe)]
#[educe(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Bn<P: BnConfig>(PhantomData<fn() -> P>);

impl<P: BnConfig> Bn<P> {
    /// Evaluates the line function at point p.
    fn ell(f: &mut Fp12<P::Fp12Config>, coeffs: &g2::EllCoeff<P>, p: &G1Affine<P>) {
        let mut c0 = coeffs.0;
        let mut c1 = coeffs.1;
        let mut c2 = coeffs.2;

        match P::TWIST_TYPE {
            TwistType::M => {
                c2.mul_assign_by_fp(&p.y);
                c1.mul_assign_by_fp(&p.x);
                f.mul_by_014(&c0, &c1, &c2);
            },
            TwistType::D => {
                c0.mul_assign_by_fp(&p.y);
                c1.mul_assign_by_fp(&p.x);
                f.mul_by_034(&c0, &c1, &c2);
            },
        }
    }

    fn exp_by_neg_x(mut f: Fp12<P::Fp12Config>) -> Fp12<P::Fp12Config> {
        f = f.cyclotomic_exp(P::X);
        if !P::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }
        f
    }

    fn exp_by_neg_x_non_cyclotomic(mut f: Fp12<P::Fp12Config>) -> Fp12<P::Fp12Config> {
        f = f.pow(P::X);
        if !P::X_IS_NEGATIVE {
            f.inverse_in_place();
        }
        f
    }
}

impl<P: BnConfig> Pairing for Bn<P> {
    type BaseField = <P::G1Config as CurveConfig>::BaseField;
    type ScalarField = <P::G1Config as CurveConfig>::ScalarField;
    type G1 = G1Projective<P>;
    type G1Affine = G1Affine<P>;
    type G1Prepared = G1Prepared<P>;
    type G2 = G2Projective<P>;
    type G2Affine = G2Affine<P>;
    type G2Prepared = G2Prepared<P>;
    type TargetField = Fp12<P::Fp12Config>;

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> MillerLoopOutput<Self> {
        P::multi_miller_loop(a, b)
    }

    fn final_exponentiation(f: MillerLoopOutput<Self>) -> Option<PairingOutput<Self>> {
        P::final_exponentiation(f)
    }
}

impl<P: BnConfig> CompressedPairing for Bn<P> {
    type CompressedTargetField = P::CompressedFp12Config;

    fn compressed_final_exponentiation(
        f: MillerLoopOutput<Self>,
    ) -> Option<Self::CompressedTargetField> {
        P::compressed_final_exponentiation(f)
    }
}
