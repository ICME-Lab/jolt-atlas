//! Fq12 polynomial operations and conversions for BN254
use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, Zero};

/// Convert Fq12 to polynomial representation using tower basis mapping
///
/// Maps Fq12 basis elements to powers of w:
/// - (c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2) → (w^0, w^2, w^4, w^1, w^3, w^5)
/// - Applies the mapping: (x + y·u)·w^k = (x - 9y)·w^k + y·w^{k+6}
pub fn fq12_to_poly12_coeffs(a: &Fq12) -> [Fq; 12] {
    // Tower basis element mappings: (outer_idx, inner_idx, w_power)
    const MAPPINGS: [(usize, usize, usize); 6] = [
        (0, 0, 0), // a.c0.c0 → w^0
        (0, 1, 2), // a.c0.c1 → w^2
        (0, 2, 4), // a.c0.c2 → w^4
        (1, 0, 1), // a.c1.c0 → w^1
        (1, 1, 3), // a.c1.c1 → w^3
        (1, 2, 5), // a.c1.c2 → w^5
    ];

    let nine = Fq::from(9);
    let mut coeffs = [Fq::zero(); 12];

    for &(outer, inner, w_power) in &MAPPINGS {
        let fp2 = match (outer, inner) {
            (0, 0) => &a.c0.c0,
            (0, 1) => &a.c0.c1,
            (0, 2) => &a.c0.c2,
            (1, 0) => &a.c1.c0,
            (1, 1) => &a.c1.c1,
            (1, 2) => &a.c1.c2,
            _ => unreachable!(),
        };

        let (x, y) = (fp2.c0, fp2.c1);
        // Apply: (x + y·u)·w^k = (x - 9y)·w^k + y·w^{k+6}
        coeffs[w_power] += x - nine * y;
        coeffs[w_power + 6] += y;
    }

    coeffs
}

/// Coefficients for the minimal polynomial g(X) = X^12 - 18 X^6 + 82
const G_COEFF_0: u64 = 82;
const G_COEFF_6: u64 = 18;

/// Evaluate g(X) = X^12 - 18 X^6 + 82 at a given point r
pub fn g_eval(r: &Fq) -> Fq {
    let r6 = (r.square() * r).square(); // r^6 = (r^2 * r)^2
    let r12 = r6.square();
    r12 - Fq::from(G_COEFF_6) * r6 + Fq::from(G_COEFF_0)
}

/// Horner evaluation for arbitrary-degree poly
pub fn eval_poly_vec(coeffs: &[Fq], r: &Fq) -> Fq {
    coeffs.iter().rev().fold(Fq::zero(), |acc, c| acc * r + c)
}

/// Build the coefficients for g(X) = X^12 - 18 X^6 + 82
pub fn g_coeffs() -> Vec<Fq> {
    let mut g = vec![Fq::zero(); 13];
    g[0] = Fq::from(G_COEFF_0);
    g[6] = -Fq::from(G_COEFF_6);
    g[12] = Fq::one();
    g
}

/// Compute the multilinear extension (MLE) of a univariate polynomial.
pub fn to_multilinear_evals(coeffs: &[Fq; 12]) -> Vec<Fq> {
    // Evaluate polynomial at points 0..16
    (0..16)
        .map(|i| {
            let x = Fq::from(i as u64);
            eval_poly_vec(&coeffs[..], &x)
        })
        .collect()
}

/// Convert Fq12 element to multilinear extension evaluations.
/// First converts to polynomial representation, then computes MLE.
pub fn fq12_to_multilinear_evals(a: &Fq12) -> Vec<Fq> {
    to_multilinear_evals(&fq12_to_poly12_coeffs(a))
}

/// Evaluate a multilinear polynomial at a given point.
pub fn eval_multilinear(evals: &[Fq], point: &[Fq]) -> Fq {
    let n = point.len();
    assert_eq!(
        evals.len(),
        1 << n,
        "Number of evaluations must be 2^n where n is dimension"
    );

    let mut result = Fq::zero();
    for (i, &eval) in evals.iter().enumerate() {
        let mut term = eval;
        for j in 0..n {
            let bit = (i >> j) & 1;
            term *= if bit == 1 {
                point[j]
            } else {
                Fq::one() - point[j]
            };
        }
        result += term;
    }
    result
}

/// Compute equality function weights eq(z, x) for all x ∈ {0,1}^4
pub fn eq_weights(z: &[Fq]) -> Vec<Fq> {
    assert_eq!(z.len(), 4, "Point z must be 4-dimensional");
    let mut w = vec![Fq::zero(); 16];

    for idx in 0..16 {
        // Binary decomposition of idx
        let x0 = Fq::from((idx & 1) as u64);
        let x1 = Fq::from(((idx >> 1) & 1) as u64);
        let x2 = Fq::from(((idx >> 2) & 1) as u64);
        let x3 = Fq::from(((idx >> 3) & 1) as u64);

        // eq(z, x) = ∏ᵢ ((1-zᵢ)(1-xᵢ) + zᵢxᵢ)
        let t0 = (Fq::one() - z[0]) * (Fq::one() - x0) + z[0] * x0;
        let t1 = (Fq::one() - z[1]) * (Fq::one() - x1) + z[1] * x1;
        let t2 = (Fq::one() - z[2]) * (Fq::one() - x2) + z[2] * x2;
        let t3 = (Fq::one() - z[3]) * (Fq::one() - x3) + z[3] * x3;

        w[idx] = t0 * t1 * t2 * t3;
    }

    w
}
