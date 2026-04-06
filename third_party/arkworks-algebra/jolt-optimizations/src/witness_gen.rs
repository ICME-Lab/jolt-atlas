use crate::fq12_poly::{eq_weights, eval_multilinear, fq12_to_multilinear_evals, g_coeffs};
use ark_bn254::{Fq, Fq12, Fr};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// square-and-multiply witness generation
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct ExponentiationSteps {
    pub base: Fq12,                  // A (base)
    pub exponent: Fr,                // e (exponent)
    pub result: Fq12,                // Final result A^e
    pub rho_mles: Vec<Vec<Fq>>,      // MLEs of ρ_0, ρ_1, ..., ρ_t
    pub quotient_mles: Vec<Vec<Fq>>, // MLEs of Q_1, Q_2, ..., Q_t
    pub bits: Vec<bool>,             // b_1, b_2, ..., b_t
}

impl ExponentiationSteps {
    /// Generate MLE witness for base^exponent using MSB-first square-and-multiply
    pub fn new(base: Fq12, exponent: Fr) -> Self {
        let bits_le = exponent.into_bigint().to_bits_le();

        let msb_idx = match bits_le.iter().rposition(|&b| b) {
            None => {
                return Self {
                    base,
                    exponent,
                    result: Fq12::one(),
                    rho_mles: vec![fq12_to_multilinear_evals(&Fq12::one())], // ρ_0 = 1
                    quotient_mles: vec![],
                    bits: vec![],
                };
            },
            Some(i) => i,
        };

        // Special case: exponent == 1 ⇒ result = base
        if msb_idx == 0 && bits_le[0] {
            return Self {
                base,
                exponent,
                result: base,
                rho_mles: vec![
                    fq12_to_multilinear_evals(&Fq12::one()), // ρ_0
                    fq12_to_multilinear_evals(&base),        // ρ_1
                ],
                quotient_mles: vec![],
                bits: vec![true],
            };
        }

        let bits_msb: Vec<bool> = (0..=msb_idx).rev().map(|i| bits_le[i]).collect();

        let mut rho = Fq12::one();
        let mut rho_mles = vec![fq12_to_multilinear_evals(&rho)];
        let mut quotient_mles = vec![];
        let mut bits = vec![];

        for &b in &bits_msb {
            bits.push(b);

            let rho_prev = rho; // ρ_{i-1}
            let rho_sq = rho_prev.square(); // ρ_{i-1}²
            let rho_i = if b { rho_sq * base } else { rho_sq }; // ρ_i

            // One quotient per step for: ρ_i(X) - ρ_{i-1}(X)² * A(X)^{b} = Q_i(X) g(X)
            let q_i = compute_step_quotient_msb(rho_prev, rho_i, base, b);
            quotient_mles.push(q_i);

            rho = rho_i;
            rho_mles.push(fq12_to_multilinear_evals(&rho));
        }

        Self {
            base,
            exponent,
            result: rho,
            rho_mles,
            quotient_mles,
            bits,
        }
    }

    /// Verify that the final result matches base^exponent
    pub fn verify_result(&self) -> bool {
        self.result == self.base.pow(self.exponent.into_bigint())
    }

    /// Verify constraint at a Boolean cube point
    /// Checks that the constraint holds at cube vertices where it was constructed to be zero
    pub fn verify_constraint_at_cube_point(&self, step: usize, cube_index: usize) -> bool {
        if step == 0 || step > self.quotient_mles.len() || cube_index >= 16 {
            return false;
        }
        let point = index_to_boolean_point(cube_index);

        // Evaluate MLEs
        let rho_prev = eval_mle_at_boolean_point(&self.rho_mles[step - 1], &point);
        let rho_curr = eval_mle_at_boolean_point(&self.rho_mles[step], &point);
        let quotient = eval_mle_at_boolean_point(&self.quotient_mles[step - 1], &point);

        let base_mle = fq12_to_multilinear_evals(&self.base);
        let base_eval = eval_mle_at_boolean_point(&base_mle, &point);
        let g_mle = get_g_mle();
        let g_eval = eval_mle_at_boolean_point(&g_mle, &point);

        // Compute constraint: ρ_i - ρ_{i-1}² * base^{b_i} - Q_i * g
        let bit = self.bits[step - 1];
        let base_power = if bit { base_eval } else { Fq::one() };
        let constraint = rho_curr - rho_prev.square() * base_power - quotient * g_eval;
        constraint.is_zero()
    }

    pub fn num_steps(&self) -> usize {
        self.quotient_mles.len()
    }
}

/// Compute quotient MLE
fn compute_step_quotient_msb(rho_prev: Fq12, rho_i: Fq12, base: Fq12, bit: bool) -> Vec<Fq> {
    let rho_prev_mle = fq12_to_multilinear_evals(&rho_prev);
    let rho_i_mle = fq12_to_multilinear_evals(&rho_i);
    let base_mle = fq12_to_multilinear_evals(&base);

    let g_mle = get_g_mle();

    // Compute the quotient MLE pointwise: Q_i(x) = (ρ_i(x) - ρ_{i-1}(x)² * base(x)^{b_i}) / g(x)
    let mut quotient_mle = vec![Fq::zero(); 16];
    for j in 0..16 {
        let rho_prev_sq = rho_prev_mle[j].square();
        let base_power = if bit { base_mle[j] } else { Fq::one() };
        let expected = rho_prev_sq * base_power;

        // Q_i(x) = (ρ_i(x) - expected) / g(x)
        if !g_mle[j].is_zero() {
            quotient_mle[j] = (rho_i_mle[j] - expected) / g_mle[j];
        }
    }

    quotient_mle
}

/// Get g as MLE evaluations over the Boolean cube {0,1}^4
pub fn get_g_mle() -> Vec<Fq> {
    use crate::fq12_poly::eval_poly_vec;
    let g_vec = g_coeffs();

    (0..16)
        .map(|i| {
            let x = Fq::from(i as u64);
            eval_poly_vec(&g_vec[..], &x)
        })
        .collect()
}

/// Convert a cube index (0..15) to a Boolean point in {0,1}^4
pub(crate) fn index_to_boolean_point(index: usize) -> Vec<Fq> {
    vec![
        Fq::from((index & 1) as u64),        // bit 0
        Fq::from(((index >> 1) & 1) as u64), // bit 1
        Fq::from(((index >> 2) & 1) as u64), // bit 2
        Fq::from(((index >> 3) & 1) as u64), // bit 3
    ]
}

/// Evaluate an MLE at a Boolean cube point
/// For Boolean points, this is equivalent to indexing but makes the evaluation explicit
fn eval_mle_at_boolean_point(mle: &[Fq], point: &[Fq]) -> Fq {
    // For Boolean points, we could just index, but using eval_multilinear
    // makes it clear we're doing MLE evaluation
    eval_multilinear(mle, point)
}

/// H(x) = ρᵢ(x) - ρᵢ₋₁(x)² · A(x)^{bᵢ} - Qᵢ(x) · g(x) for x ∈ {0,1}^4
/// H̃(z) = Σ_{x∈{0,1}^4} eq(z,x) · H(x)
pub fn h_tilde_at_point(
    rho_prev_mle: &[Fq],
    rho_curr_mle: &[Fq],
    base_mle: &[Fq],
    q_mle: &[Fq],
    g_mle: &[Fq],
    bit: bool,
    z: &[Fq],
) -> Fq {
    assert_eq!(rho_prev_mle.len(), 16);
    assert_eq!(rho_curr_mle.len(), 16);
    assert_eq!(base_mle.len(), 16);
    assert_eq!(q_mle.len(), 16);
    assert_eq!(g_mle.len(), 16);
    assert_eq!(z.len(), 4);

    let w = eq_weights(z);
    let mut acc = Fq::zero();

    for j in 0..16 {
        // Compute H(x_j) where x_j is the j-th hypercube vertex
        let prod = rho_prev_mle[j].square() * if bit { base_mle[j] } else { Fq::one() };
        let h_x = rho_curr_mle[j] - prod - q_mle[j] * g_mle[j];

        acc += h_x * w[j];
    }

    acc
}
