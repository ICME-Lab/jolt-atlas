//! Succinct Verifier R1CS Circuit
//!
//! This module encodes the sumcheck verifier's checks as R1CS constraints,
//! enabling efficient verification through NIFS folding.
//!
//! # Verifier Circuit Structure
//!
//! The sumcheck verifier performs O(log n) checks:
//! 1. For each round i: h_i(0) + h_i(1) = claim_{i-1}
//! 2. For each round i: h_i(r_i) = claim_i (next round's claim)
//!
//! These checks are encoded as R1CS constraints that can be efficiently
//! verified through the Nova/NIFS folding scheme.
//!
//! # Variable Layout
//!
//! The R1CS uses the following variable indexing:
//! - Index 0: Constant 1
//! - Index 1: Initial claim (public)
//! - Index 2..2+n: Challenges r_1, ..., r_n (public)
//! - Index 2+n: Final expected claim (public)
//! - Index 2+n+1..: Private witness variables:
//!   - Round polynomial coefficients c_0, c_1, ..., c_d for each round
//!   - Intermediate claims for each round
//!   - Powers of challenges r^2, r^3, ..., r^d for polynomial evaluation
//!   - Product terms t_i = c_i * r^i for each round (for proper evaluation encoding)

use jolt_core::field::JoltField;

use super::relaxed_r1cs::R1CSMatrices;

/// The sumcheck verifier circuit encoded as R1CS.
///
/// This circuit verifies that the sumcheck protocol was executed correctly
/// by encoding the verifier's checks as constraints.
#[derive(Clone, Debug)]
pub struct VerifierR1CSCircuit<F: JoltField> {
    /// The number of sumcheck rounds
    num_rounds: usize,
    /// The maximum degree of the sumcheck polynomial
    degree: usize,
    /// The R1CS matrices encoding the verifier's checks
    pub matrices: R1CSMatrices<F>,
    /// Variable indices for accessing circuit variables
    pub var_indices: VariableIndices,
}

/// Indices for accessing variables in the R1CS circuit.
#[derive(Clone, Debug)]
pub struct VariableIndices {
    /// Index of constant 1
    pub const_one: usize,
    /// Index of initial claim
    pub initial_claim: usize,
    /// Starting index of challenges
    pub challenges_start: usize,
    /// Index of final expected claim
    pub final_claim: usize,
    /// Starting index of private witnesses
    pub private_start: usize,
    /// Number of rounds
    pub num_rounds: usize,
    /// Polynomial degree
    pub degree: usize,
}

impl VariableIndices {
    /// Creates new variable indices for the given circuit parameters.
    pub fn new(num_rounds: usize, degree: usize) -> Self {
        Self {
            const_one: 0,
            initial_claim: 1,
            challenges_start: 2,
            final_claim: 2 + num_rounds,
            private_start: 2 + num_rounds + 1,
            num_rounds,
            degree,
        }
    }

    /// Returns the index of the i-th challenge (0-indexed).
    pub fn challenge(&self, round: usize) -> usize {
        assert!(round < self.num_rounds);
        self.challenges_start + round
    }

    /// Returns the starting index of coefficients for the given round.
    pub fn coefficients_start(&self, round: usize) -> usize {
        assert!(round < self.num_rounds);
        self.private_start + round * (self.degree + 1)
    }

    /// Returns the index of the i-th coefficient for the given round.
    pub fn coefficient(&self, round: usize, coeff_idx: usize) -> usize {
        assert!(coeff_idx <= self.degree);
        self.coefficients_start(round) + coeff_idx
    }

    /// Returns the starting index of intermediate claims.
    pub fn intermediate_claims_start(&self) -> usize {
        self.private_start + self.num_rounds * (self.degree + 1)
    }

    /// Returns the index of the intermediate claim after round i.
    /// For round 0, the claim before is initial_claim (public).
    /// For round n-1, the claim after is final_claim (public).
    pub fn intermediate_claim(&self, round: usize) -> usize {
        assert!(round < self.num_rounds - 1);
        self.intermediate_claims_start() + round
    }

    /// Returns the starting index of challenge powers for the given round.
    /// Powers are r^2, r^3, ..., r^d (r^1 = r is already available).
    pub fn powers_start(&self, round: usize) -> usize {
        let claims_end = self.intermediate_claims_start() + (self.num_rounds - 1);
        claims_end + round * self.degree.saturating_sub(1)
    }

    /// Returns the index of r^power for the given round.
    pub fn power(&self, round: usize, power: usize) -> usize {
        assert!(power >= 2 && power <= self.degree);
        self.powers_start(round) + (power - 2)
    }

    /// Returns the starting index of product terms for the given round.
    /// Products are t_1 = c_1*r, t_2 = c_2*r^2, ..., t_d = c_d*r^d
    pub fn products_start(&self, round: usize) -> usize {
        let powers_end = self.powers_start(self.num_rounds - 1)
            + self.degree.saturating_sub(1);
        powers_end + round * self.degree
    }

    /// Returns the index of product t_i = c_i * r^i for the given round.
    /// i ranges from 1 to degree.
    pub fn product(&self, round: usize, i: usize) -> usize {
        assert!(i >= 1 && i <= self.degree);
        self.products_start(round) + (i - 1)
    }
}

impl<F: JoltField> VerifierR1CSCircuit<F> {
    /// Creates a new verifier circuit for a sumcheck with the given parameters.
    ///
    /// # Arguments
    /// * `num_rounds` - Number of sumcheck rounds
    /// * `degree` - Maximum degree of the sumcheck polynomial
    pub fn new(num_rounds: usize, degree: usize) -> Self {
        let var_indices = VariableIndices::new(num_rounds, degree);
        let matrices = Self::build_matrices(num_rounds, degree, &var_indices);
        Self {
            num_rounds,
            degree,
            matrices,
            var_indices,
        }
    }

    /// Returns the number of sumcheck rounds.
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Returns the degree of the sumcheck polynomial.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Builds the R1CS matrices for the verifier circuit.
    ///
    /// The circuit encodes the following constraints:
    /// 1. Sum check: h(0) + h(1) = claim_prev for each round
    /// 2. Product constraints: t_i = c_i * r^i for each round
    /// 3. Eval check: c_0 + t_1 + t_2 + ... + t_d = claim_next for each round
    /// 4. Power constraints: r^{i+1} = r^i * r for computing polynomial evaluation
    fn build_matrices(
        num_rounds: usize,
        degree: usize,
        indices: &VariableIndices,
    ) -> R1CSMatrices<F> {
        // Calculate number of variables
        let num_public_inputs = 1 + num_rounds + 1; // initial_claim, challenges, final_claim
        let num_coeffs = num_rounds * (degree + 1);
        let num_intermediate_claims = num_rounds.saturating_sub(1);
        let num_powers = num_rounds * degree.saturating_sub(1); // r^2, ..., r^d for each round
        let num_products = num_rounds * degree; // t_1, ..., t_d for each round
        let num_private_vars = num_coeffs + num_intermediate_claims + num_powers + num_products;
        let num_variables = 1 + num_public_inputs + num_private_vars;

        // Calculate number of constraints
        // - num_rounds sum checks
        // - num_rounds * degree product constraints (t_i = c_i * r^i)
        // - num_rounds eval checks (c_0 + t_1 + ... + t_d = claim_next)
        // - num_rounds * (degree - 1) power constraints
        let num_product_constraints = num_rounds * degree;
        let num_power_constraints = num_rounds * degree.saturating_sub(1);
        let num_constraints = num_rounds + num_product_constraints + num_rounds + num_power_constraints;

        let mut matrices = R1CSMatrices::new(num_constraints, num_variables, num_public_inputs);
        let mut constraint_idx = 0;

        // Add constraints for each round
        for round in 0..num_rounds {
            // Constraint: h(0) + h(1) = claim_prev
            Self::add_sum_check_constraint(&mut matrices, constraint_idx, round, degree, indices);
            constraint_idx += 1;
        }

        // Add product constraints: t_i = c_i * r^i for each round
        for round in 0..num_rounds {
            for i in 1..=degree {
                Self::add_product_constraint(&mut matrices, constraint_idx, round, i, indices);
                constraint_idx += 1;
            }
        }

        // Add evaluation constraints: c_0 + t_1 + ... + t_d = claim_next
        for round in 0..num_rounds {
            Self::add_eval_constraint(&mut matrices, constraint_idx, round, degree, indices);
            constraint_idx += 1;
        }

        // Add power constraints: r^{i+1} = r^i * r for each round
        for round in 0..num_rounds {
            for power in 2..=degree {
                Self::add_power_constraint(&mut matrices, constraint_idx, round, power, indices);
                constraint_idx += 1;
            }
        }

        matrices
    }

    /// Adds the sum check constraint: h(0) + h(1) = claim_prev
    ///
    /// Encodes: (2*c_0 + c_1 + c_2 + ... + c_d) * 1 = claim_prev
    fn add_sum_check_constraint(
        matrices: &mut R1CSMatrices<F>,
        constraint_idx: usize,
        round: usize,
        degree: usize,
        indices: &VariableIndices,
    ) {
        // A: 2*c_0 + c_1 + c_2 + ... + c_d
        // c_0 coefficient is 2 (contributes to both h(0) and h(1))
        matrices.A.add_entry(
            constraint_idx,
            indices.coefficient(round, 0),
            F::from_u64(2u64),
        );
        // c_1, c_2, ..., c_d each contribute with coefficient 1
        for i in 1..=degree {
            matrices
                .A
                .add_entry(constraint_idx, indices.coefficient(round, i), F::one());
        }

        // B: 1 (constant)
        matrices
            .B
            .add_entry(constraint_idx, indices.const_one, F::one());

        // C: claim_prev
        // For round 0, claim_prev is initial_claim (public input)
        // For round > 0, claim_prev is intermediate_claim[round-1]
        if round == 0 {
            matrices
                .C
                .add_entry(constraint_idx, indices.initial_claim, F::one());
        } else {
            matrices.C.add_entry(
                constraint_idx,
                indices.intermediate_claim(round - 1),
                F::one(),
            );
        }
    }

    /// Adds a product constraint: t_i = c_i * r^i
    ///
    /// This is a multiplication constraint that verifies the product term.
    /// For i=1: t_1 = c_1 * r (A=c_1, B=r, C=t_1)
    /// For i>1: t_i = c_i * r^i (A=c_i, B=r^i, C=t_i)
    fn add_product_constraint(
        matrices: &mut R1CSMatrices<F>,
        constraint_idx: usize,
        round: usize,
        i: usize,
        indices: &VariableIndices,
    ) {
        // A: c_i
        matrices
            .A
            .add_entry(constraint_idx, indices.coefficient(round, i), F::one());

        // B: r^i
        if i == 1 {
            // r^1 = r (challenge, public input)
            matrices
                .B
                .add_entry(constraint_idx, indices.challenge(round), F::one());
        } else {
            // r^i (witness)
            matrices
                .B
                .add_entry(constraint_idx, indices.power(round, i), F::one());
        }

        // C: t_i (product witness)
        matrices
            .C
            .add_entry(constraint_idx, indices.product(round, i), F::one());
    }

    /// Adds the evaluation constraint: h(r) = claim_next
    ///
    /// Encodes: (c_0 + t_1 + t_2 + ... + t_d) * 1 = claim_next
    /// where t_i = c_i * r^i are auxiliary product terms verified separately.
    fn add_eval_constraint(
        matrices: &mut R1CSMatrices<F>,
        constraint_idx: usize,
        round: usize,
        degree: usize,
        indices: &VariableIndices,
    ) {
        // A: c_0 + t_1 + t_2 + ... + t_d
        // c_0 coefficient is 1
        matrices
            .A
            .add_entry(constraint_idx, indices.coefficient(round, 0), F::one());

        // Add product terms t_1, t_2, ..., t_d
        for i in 1..=degree {
            matrices
                .A
                .add_entry(constraint_idx, indices.product(round, i), F::one());
        }

        // B: 1 (constant)
        matrices
            .B
            .add_entry(constraint_idx, indices.const_one, F::one());

        // C: claim_next
        // For round < num_rounds - 1, claim_next is intermediate_claim[round]
        // For last round, claim_next is final_claim (public input)
        if round == indices.num_rounds - 1 {
            matrices
                .C
                .add_entry(constraint_idx, indices.final_claim, F::one());
        } else {
            matrices
                .C
                .add_entry(constraint_idx, indices.intermediate_claim(round), F::one());
        }
    }

    /// Adds a power constraint: r^power = r^{power-1} * r
    ///
    /// For power=2: r^2 = r * r
    /// For power>2: r^power = r^{power-1} * r
    fn add_power_constraint(
        matrices: &mut R1CSMatrices<F>,
        constraint_idx: usize,
        round: usize,
        power: usize,
        indices: &VariableIndices,
    ) {
        // A: r^{power-1}
        if power == 2 {
            // r^1 = r (challenge, public input)
            matrices
                .A
                .add_entry(constraint_idx, indices.challenge(round), F::one());
        } else {
            // r^{power-1} (witness)
            matrices
                .A
                .add_entry(constraint_idx, indices.power(round, power - 1), F::one());
        }

        // B: r (challenge, public input)
        matrices
            .B
            .add_entry(constraint_idx, indices.challenge(round), F::one());

        // C: r^power (witness)
        matrices
            .C
            .add_entry(constraint_idx, indices.power(round, power), F::one());
    }

    /// Generates a witness for the verifier circuit.
    ///
    /// # Arguments
    /// * `initial_claim` - The initial sumcheck claim
    /// * `challenges` - The challenges r_1, ..., r_n
    /// * `round_polynomials` - The coefficients for each round's polynomial
    /// * `final_claim` - The expected final evaluation
    pub fn generate_witness(
        &self,
        initial_claim: F,
        challenges: &[F],
        round_polynomials: &[Vec<F>],
        final_claim: F,
    ) -> VerifierWitness<F> {
        assert_eq!(challenges.len(), self.num_rounds);
        assert_eq!(round_polynomials.len(), self.num_rounds);

        // Compute intermediate claims from polynomial evaluations
        let mut intermediate_claims = Vec::with_capacity(self.num_rounds.saturating_sub(1));
        for round in 0..self.num_rounds - 1 {
            let coeffs = &round_polynomials[round];
            let r = challenges[round];
            let eval = Self::evaluate_polynomial(coeffs, r);
            intermediate_claims.push(eval);
        }

        // Compute powers of challenges for polynomial evaluation
        let mut powers = Vec::with_capacity(self.num_rounds);
        for round in 0..self.num_rounds {
            let r = challenges[round];
            let mut round_powers = Vec::with_capacity(self.degree.saturating_sub(1));
            let mut r_power = r * r; // Start with r^2
            for _ in 2..=self.degree {
                round_powers.push(r_power);
                r_power = r_power * r;
            }
            powers.push(round_powers);
        }

        // Compute product terms t_i = c_i * r^i for each round
        let mut products = Vec::with_capacity(self.num_rounds);
        for round in 0..self.num_rounds {
            let coeffs = &round_polynomials[round];
            let r = challenges[round];
            let mut round_products = Vec::with_capacity(self.degree);
            for i in 1..=self.degree {
                let r_power = if i == 1 {
                    r
                } else {
                    powers[round][i - 2]
                };
                let product = coeffs[i] * r_power;
                round_products.push(product);
            }
            products.push(round_products);
        }

        VerifierWitness {
            initial_claim,
            challenges: challenges.to_vec(),
            round_coefficients: round_polynomials.to_vec(),
            final_claim,
            intermediate_claims,
            powers,
            products,
        }
    }

    /// Evaluates a polynomial at a point using Horner's method.
    fn evaluate_polynomial(coeffs: &[F], x: F) -> F {
        if coeffs.is_empty() {
            return F::zero();
        }
        let mut result = *coeffs.last().unwrap();
        for coeff in coeffs.iter().rev().skip(1) {
            result = result * x + *coeff;
        }
        result
    }

    /// Verifies that a witness satisfies the verifier circuit constraints.
    ///
    /// This checks:
    /// 1. Sum check property: h(0) + h(1) = claim_prev for each round
    /// 2. Evaluation property: h(r) = claim_next for each round
    /// 3. Power constraints: r^i = r^{i-1} * r
    ///
    /// Returns true if all constraints are satisfied.
    pub fn verify_witness(&self, witness: &VerifierWitness<F>) -> bool {
        witness.verify_sum_checks()
            && witness.verify_evaluations()
            && witness.verify_powers()
    }
}

/// Witness for the verifier circuit.
#[derive(Clone, Debug)]
pub struct VerifierWitness<F: JoltField> {
    /// The initial sumcheck claim
    pub initial_claim: F,
    /// The verifier's challenges
    pub challenges: Vec<F>,
    /// Coefficients for each round's univariate polynomial
    pub round_coefficients: Vec<Vec<F>>,
    /// The expected final claim
    pub final_claim: F,
    /// Intermediate claims (evaluation results for each round except the last)
    pub intermediate_claims: Vec<F>,
    /// Powers of challenges: powers[round] = [r^2, r^3, ..., r^d]
    pub powers: Vec<Vec<F>>,
    /// Product terms: products[round] = [c_1*r, c_2*r^2, ..., c_d*r^d]
    pub products: Vec<Vec<F>>,
}

impl<F: JoltField> VerifierWitness<F> {
    /// Returns the number of rounds.
    pub fn num_rounds(&self) -> usize {
        self.challenges.len()
    }

    /// Converts the witness to a flat vector for R1CS.
    ///
    /// Layout matches the circuit's variable indices:
    /// [initial_claim, challenges..., final_claim, coefficients..., intermediate_claims..., powers..., products...]
    pub fn to_flat_vector(&self) -> Vec<F> {
        let mut result = vec![self.initial_claim];
        result.extend_from_slice(&self.challenges);
        result.push(self.final_claim);

        // Private witnesses: coefficients
        for coeffs in &self.round_coefficients {
            result.extend_from_slice(coeffs);
        }

        // Private witnesses: intermediate claims
        result.extend_from_slice(&self.intermediate_claims);

        // Private witnesses: powers of challenges
        for round_powers in &self.powers {
            result.extend_from_slice(round_powers);
        }

        // Private witnesses: product terms
        for round_products in &self.products {
            result.extend_from_slice(round_products);
        }

        result
    }

    /// Verifies that the witness satisfies the sum check property for each round.
    ///
    /// h(0) + h(1) = claim_prev
    pub fn verify_sum_checks(&self) -> bool {
        for round in 0..self.challenges.len() {
            let coeffs = &self.round_coefficients[round];

            // h(0) = c_0
            let h_0 = coeffs[0];

            // h(1) = c_0 + c_1 + c_2 + ... + c_d
            let h_1: F = coeffs.iter().fold(F::zero(), |acc, c| acc + *c);

            // claim_prev
            let claim_prev = if round == 0 {
                self.initial_claim
            } else {
                self.intermediate_claims[round - 1]
            };

            // Check: h(0) + h(1) = claim_prev
            if h_0 + h_1 != claim_prev {
                return false;
            }
        }
        true
    }

    /// Verifies that the intermediate claims are correct polynomial evaluations.
    pub fn verify_evaluations(&self) -> bool {
        for round in 0..self.challenges.len() {
            let coeffs = &self.round_coefficients[round];
            let r = self.challenges[round];

            // Evaluate h(r) using the product terms
            let mut eval = coeffs[0]; // c_0
            for i in 1..coeffs.len() {
                eval = eval + self.products[round][i - 1];
            }

            // Check against expected claim
            let expected_claim = if round == self.challenges.len() - 1 {
                self.final_claim
            } else {
                self.intermediate_claims[round]
            };

            if eval != expected_claim {
                return false;
            }

            // Also verify product terms are correct: t_i = c_i * r^i
            for i in 1..coeffs.len() {
                let r_power = if i == 1 {
                    r
                } else {
                    self.powers[round][i - 2]
                };
                let expected_product = coeffs[i] * r_power;
                if self.products[round][i - 1] != expected_product {
                    return false;
                }
            }
        }
        true
    }

    /// Verifies that the power terms are correct: r^i = r^{i-1} * r
    pub fn verify_powers(&self) -> bool {
        for round in 0..self.challenges.len() {
            let r = self.challenges[round];

            for (i, &power_val) in self.powers[round].iter().enumerate() {
                let power = i + 2; // powers are r^2, r^3, ...
                let expected = if power == 2 {
                    r * r
                } else {
                    self.powers[round][i - 1] * r
                };
                if power_val != expected {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_verifier_circuit_creation() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(10, 3);
        assert_eq!(circuit.num_rounds(), 10);
        assert_eq!(circuit.degree(), 3);
    }

    #[test]
    fn test_variable_indices() {
        let indices = VariableIndices::new(3, 2);

        assert_eq!(indices.const_one, 0);
        assert_eq!(indices.initial_claim, 1);
        assert_eq!(indices.challenges_start, 2);
        assert_eq!(indices.challenge(0), 2);
        assert_eq!(indices.challenge(1), 3);
        assert_eq!(indices.challenge(2), 4);
        assert_eq!(indices.final_claim, 5);
        assert_eq!(indices.private_start, 6);

        // Coefficients for round 0: indices 6, 7, 8 (c_0, c_1, c_2)
        assert_eq!(indices.coefficient(0, 0), 6);
        assert_eq!(indices.coefficient(0, 1), 7);
        assert_eq!(indices.coefficient(0, 2), 8);

        // Coefficients for round 1: indices 9, 10, 11
        assert_eq!(indices.coefficient(1, 0), 9);
    }

    #[test]
    fn test_witness_generation() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(3, 2);

        let initial_claim = Fr::from(100u64);
        let challenges = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let round_polys = vec![
            vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)],
            vec![Fr::from(40u64), Fr::from(50u64), Fr::from(60u64)],
            vec![Fr::from(70u64), Fr::from(80u64), Fr::from(90u64)],
        ];
        let final_claim = Fr::from(42u64);

        let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);

        assert_eq!(witness.num_rounds(), 3);
        assert_eq!(witness.initial_claim, initial_claim);
        assert_eq!(witness.final_claim, final_claim);
        assert_eq!(witness.intermediate_claims.len(), 2); // 3 rounds - 1
        assert_eq!(witness.powers.len(), 3); // one per round
        assert_eq!(witness.products.len(), 3); // one per round
    }

    #[test]
    fn test_polynomial_evaluation() {
        // Test h(x) = 1 + 2x + 3x^2 at x = 2
        // h(2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let coeffs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let x = Fr::from(2u64);
        let result = VerifierR1CSCircuit::evaluate_polynomial(&coeffs, x);
        assert_eq!(result, Fr::from(17u64));
    }

    #[test]
    fn test_witness_with_valid_sumcheck() {
        // Create a valid sumcheck witness where h(0) + h(1) = claim
        // For h(x) = c_0 + c_1*x, we have:
        // h(0) = c_0
        // h(1) = c_0 + c_1
        // h(0) + h(1) = 2*c_0 + c_1

        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(2, 1);

        // Round 1: h(x) = 30 + 40x
        // h(0) + h(1) = 30 + 70 = 100 = initial_claim
        // h(r=2) = 30 + 80 = 110 = intermediate_claim_0

        // Round 2: h(x) = 50 + 10x
        // h(0) + h(1) = 50 + 60 = 110 = intermediate_claim_0
        // h(r=3) = 50 + 30 = 80 = final_claim

        let initial_claim = Fr::from(100u64);
        let challenges = vec![Fr::from(2u64), Fr::from(3u64)];
        let round_polys = vec![
            vec![Fr::from(30u64), Fr::from(40u64)],
            vec![Fr::from(50u64), Fr::from(10u64)],
        ];
        let final_claim = Fr::from(80u64);

        let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);

        // Verify intermediate claim
        assert_eq!(witness.intermediate_claims.len(), 1);
        assert_eq!(witness.intermediate_claims[0], Fr::from(110u64));

        // Verify sum checks
        assert!(witness.verify_sum_checks());

        // Verify evaluations
        assert!(witness.verify_evaluations());

        // Verify powers
        assert!(witness.verify_powers());
    }

    #[test]
    fn test_sum_check_constraint_encoding() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(2, 2);

        // Check that the constraint count is correct
        // 2 rounds * (1 sum + 2 products + 1 eval + 1 power) = 2 * 5 = 10
        // Actually: 2 sum + 2*2 products + 2 eval + 2*1 powers = 2 + 4 + 2 + 2 = 10
        let expected_constraints = 2 + 2 * 2 + 2 + 2 * 1;
        assert_eq!(circuit.matrices.A.num_rows, expected_constraints);
    }

    #[test]
    fn test_power_constraints() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(1, 3);

        // For degree 3, we need r^2 and r^3
        // Power constraints: r^2 = r * r, r^3 = r^2 * r
        // So we have 2 power constraints per round

        // Constraints: 1 sum check + 3 product checks + 1 eval check + 2 power checks = 7
        assert_eq!(circuit.matrices.A.num_rows, 7);
    }

    #[test]
    fn test_product_terms_are_correct() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(2, 2);

        let initial_claim = Fr::from(100u64);
        let challenges = vec![Fr::from(3u64), Fr::from(5u64)];
        let round_polys = vec![
            vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)],
            vec![Fr::from(40u64), Fr::from(50u64), Fr::from(60u64)],
        ];
        let final_claim = Fr::from(42u64);

        let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);

        // Round 0 with r=3:
        // t_1 = c_1 * r = 20 * 3 = 60
        // t_2 = c_2 * r^2 = 30 * 9 = 270
        assert_eq!(witness.products[0][0], Fr::from(60u64));
        assert_eq!(witness.products[0][1], Fr::from(270u64));

        // Round 1 with r=5:
        // t_1 = c_1 * r = 50 * 5 = 250
        // t_2 = c_2 * r^2 = 60 * 25 = 1500
        assert_eq!(witness.products[1][0], Fr::from(250u64));
        assert_eq!(witness.products[1][1], Fr::from(1500u64));
    }
}
