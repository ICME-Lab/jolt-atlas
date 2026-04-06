use ark_ec::bn::FromPsi6Pow;
use ark_ff::vec::Vec;
use ark_ff::{AdditiveGroup, Field, Fp12, Fp12Config, Fp6Config, MontFp};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::bn254::{Config, Fq, Fq2, Fq6, Fq6Config};

pub type Fq12 = Fp12<Fq12Config>;
pub type CompressibleFq12 = Fp12<CompressibleFq12Config>;

static Q: [u64; 8] = [
    0x3b5458a2275d69b1,
    0xa602072d09eac101,
    0x4a50189c6d96cadc,
    0x04689e957a1242c8,
    0x26edfa5c34c6b38d,
    0xb00b855116375606,
    0x599a6f7c0348d21c,
    0x925c4b8763cbf9c,
];

// https://eprint.iacr.org/2007/429.pdf Proposition 1
#[derive(Clone, Copy, CanonicalSerialize, CanonicalDeserialize)]
pub struct CompressedFq12(pub (Fq2, Fq2));

#[inline]
pub fn torus_compress_fq6(element: Fq6) -> CompressedFq12 {
    CompressedFq12((element.c0, element.c1))
}

#[inline]
pub fn torus_decompress_fq6(element: CompressedFq12) -> Fq6 {
    let c2 = (Fq2::from(3) * element.0 .0.square() + Fq6Config::NONRESIDUE)
        * (Fq2::from(3) * element.0 .1 * Fq6Config::NONRESIDUE)
            .inverse()
            .unwrap_or_else(|| panic!("c1 cannot be zero for an element with norm 1."));
    Fq6 {
        c0: element.0 .0,
        c1: element.0 .1,
        c2,
    }
}

pub fn torus_compress_psi_6_pow_to_two_fq2(element: CompressibleFq12) -> CompressedFq12 {
    let c1 = element.c0 / element.c1;
    let c1_pow = -c1.pow(Q);
    let compressed_prod = CompressibleFq12::mul_torus_compressed_elements(c1_pow, c1);
    CompressedFq12((compressed_prod.c0, compressed_prod.c1))
}

#[inline]
pub fn fq12_to_compressible_fq12(value: Fq12) -> CompressibleFq12 {
    // Divide by the generator of Fq6
    let new_c1 = Fq6 {
        c0: value.c1.c1,
        c1: value.c1.c2,
        c2: value.c1.c0 * Fq6Config::NONRESIDUE.inverse().unwrap(),
    };

    CompressibleFq12 {
        c0: value.c0,
        c1: new_c1,
    }
}

#[inline]
pub fn compressible_fq12_to_fq12(value: CompressibleFq12) -> Fq12 {
    // Multiply by the generator of Fq6
    let new_c1 = Fq6 {
        c0: value.c1.c2 * Fq6Config::NONRESIDUE,
        c1: value.c1.c0,
        c2: value.c1.c1,
    };

    Fq12 {
        c0: value.c0,
        c1: new_c1,
    }
}

static COMPRESSIBLE_FROBENIUS_COEFFS: [Fq2; 4] = [
    Fq2::new(Fq::ONE, Fq::ZERO),
    Fq2::new(
        MontFp!("2821565182194536844548159561693502659359617185244120367078079554186484126554"),
        MontFp!("3505843767911556378687030309984248845540243509899259641013678093033130930403"),
    ),
    Fq2::new(
        MontFp!("21888242871839275222246405745257275088696311157297823662689037894645226208582"),
        MontFp!("0"),
    ),
    Fq2::new(
        MontFp!("19066677689644738377698246183563772429336693972053703295610958340458742082029"),
        MontFp!("18382399103927718843559375435273026243156067647398564021675359801612095278180"),
    ),
];

#[derive(Clone, Copy)]
pub struct Fq12Config;

// Implement the compression method in Proposition 1 of https://eprint.iacr.org/2007/429.pdf.

#[derive(Clone, Copy)]
pub struct CompressibleFq12Config;

impl Fp12Config for CompressibleFq12Config {
    type Fp6Config = Fq6Config;

    // The 12th degree extension is generated as a quadratic extension over the 6th degree extension. Another way to think about this is that the field as a 12th deg extension over the base field is really the composite field of a quadratic extension and a cubic extension, with generators sqrt(\gamma) and cbrt(\gamma), respectively, where \gamma is a sextic non-residue in the base field (itself a second deg extension over the base field on which the bn254 curve is defined). Therefore, the quadratic non-residue that generates the 12th degree extension over the 6th deg base field is \gamma = Fq6::non_residue.
    const NONRESIDUE: Fq6 = Fq6::new(<Fq6Config as Fp6Config>::NONRESIDUE, Fq2::ZERO, Fq2::ZERO);

    const FROBENIUS_COEFF_FP12_C1: &'static [Fq2] = &[
        COMPRESSIBLE_FROBENIUS_COEFFS[0],
        COMPRESSIBLE_FROBENIUS_COEFFS[1],
        COMPRESSIBLE_FROBENIUS_COEFFS[2],
        COMPRESSIBLE_FROBENIUS_COEFFS[3],
        COMPRESSIBLE_FROBENIUS_COEFFS[0],
        COMPRESSIBLE_FROBENIUS_COEFFS[1],
        COMPRESSIBLE_FROBENIUS_COEFFS[2],
        COMPRESSIBLE_FROBENIUS_COEFFS[3],
        COMPRESSIBLE_FROBENIUS_COEFFS[0],
        COMPRESSIBLE_FROBENIUS_COEFFS[1],
        COMPRESSIBLE_FROBENIUS_COEFFS[2],
        COMPRESSIBLE_FROBENIUS_COEFFS[3],
    ];

    fn mul_fp6_by_nonresidue_in_place(fe: &mut Fq6) -> &mut Fq6 {
        Fq6Config::mul_fp2_by_nonresidue_in_place(&mut fe.c0);
        Fq6Config::mul_fp2_by_nonresidue_in_place(&mut fe.c1);
        Fq6Config::mul_fp2_by_nonresidue_in_place(&mut fe.c2);
        fe
    }
}

impl FromPsi6Pow<Config> for CompressedFq12 {
    fn from_psi_six_pow(value: Fq12) -> Option<Self> {
        // Reference: https://eprint.iacr.org/2007/429.pdf p.10 Proposition 1
        let compressible_value = fq12_to_compressible_fq12(value);
        Some(torus_compress_psi_6_pow_to_two_fq2(compressible_value))
    }
}

impl CompressedFq12 {
    #[inline]
    pub fn decompress_to_fq12(self) -> Fq12 {
        compressible_fq12_to_fq12(self.decompress())
    }

    #[inline]
    pub fn decompress(self) -> CompressibleFq12 {
        // https://eprint.iacr.org/2007/429.pdf p.10 equation (6)
        let c2 = (Fq2::from(3) * self.0 .0.square() + Fq6Config::NONRESIDUE)
            * (Fq2::from(3) * self.0 .1 * Fq6Config::NONRESIDUE)
                .inverse()
                .unwrap();
        CompressibleFq12::torus_decompress(Fq6 {
            c0: self.0 .0,
            c1: self.0 .1,
            c2,
        })
    }
}

impl Fp12Config for Fq12Config {
    type Fp6Config = Fq6Config;

    const NONRESIDUE: Fq6 = Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO);

    const FROBENIUS_COEFF_FP12_C1: &'static [Fq2] = &[
        // Fq6::NONRESIDUE^(((q^0) - 1) / 6)
        Fq2::new(Fq::ONE, Fq::ZERO),
        // Fq6::NONRESIDUE^(((q^1) - 1) / 6)
        Fq2::new(
            MontFp!("8376118865763821496583973867626364092589906065868298776909617916018768340080"),
            MontFp!(
                "16469823323077808223889137241176536799009286646108169935659301613961712198316"
            ),
        ),
        // Fq6::NONRESIDUE^(((q^2) - 1) / 6)
        Fq2::new(
            MontFp!(
                "21888242871839275220042445260109153167277707414472061641714758635765020556617"
            ),
            Fq::ZERO,
        ),
        // Fq6::NONRESIDUE^(((q^3) - 1) / 6)
        Fq2::new(
            MontFp!(
                "11697423496358154304825782922584725312912383441159505038794027105778954184319"
            ),
            MontFp!("303847389135065887422783454877609941456349188919719272345083954437860409601"),
        ),
        // Fq6::NONRESIDUE^(((q^4) - 1) / 6)
        Fq2::new(
            MontFp!(
                "21888242871839275220042445260109153167277707414472061641714758635765020556616"
            ),
            Fq::ZERO,
        ),
        // Fq6::NONRESIDUE^(((q^5) - 1) / 6)
        Fq2::new(
            MontFp!("3321304630594332808241809054958361220322477375291206261884409189760185844239"),
            MontFp!("5722266937896532885780051958958348231143373700109372999374820235121374419868"),
        ),
        // Fq6::NONRESIDUE^(((q^6) - 1) / 6)
        Fq2::new(MontFp!("-1"), Fq::ZERO),
        // Fq6::NONRESIDUE^(((q^7) - 1) / 6)
        Fq2::new(
            MontFp!(
                "13512124006075453725662431877630910996106405091429524885779419978626457868503"
            ),
            MontFp!("5418419548761466998357268504080738289687024511189653727029736280683514010267"),
        ),
        // Fq6::NONRESIDUE^(((q^8) - 1) / 6)
        Fq2::new(
            MontFp!("2203960485148121921418603742825762020974279258880205651966"),
            Fq::ZERO,
        ),
        // Fq6::NONRESIDUE^(((q^9) - 1) / 6)
        Fq2::new(
            MontFp!(
                "10190819375481120917420622822672549775783927716138318623895010788866272024264"
            ),
            MontFp!(
                "21584395482704209334823622290379665147239961968378104390343953940207365798982"
            ),
        ),
        // Fq6::NONRESIDUE^(((q^10) - 1) / 6)
        Fq2::new(
            MontFp!("2203960485148121921418603742825762020974279258880205651967"),
            Fq::ZERO,
        ),
        // Fq6::NONRESIDUE^(((q^11) - 1) / 6)
        Fq2::new(
            MontFp!(
                "18566938241244942414004596690298913868373833782006617400804628704885040364344"
            ),
            MontFp!(
                "16165975933942742336466353786298926857552937457188450663314217659523851788715"
            ),
        ),
    ];
}
