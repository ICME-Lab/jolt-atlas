//! BlindFold Zero-Knowledge Protocol
//!
//! This module implements the BlindFold protocol for making sumcheck proofs
//! zero-knowledge. Instead of revealing round polynomial coefficients, the prover
//! sends Pedersen commitments. Sumcheck verification is encoded into a small R1CS
//! circuit and proved via Nova folding + Spartan.

pub mod folding;
pub mod layout;
pub mod output_constraint;
#[cfg(feature = "zk")]
pub mod protocol;
pub mod r1cs;
pub mod relaxed_r1cs;
#[cfg(feature = "zk")]
pub mod spartan;
pub mod witness;

pub use folding::{commit_cross_term_rows, compute_cross_term, sample_random_satisfying_pair};
pub use r1cs::{SparseR1CSMatrix, VerifierR1CS, VerifierR1CSBuilder};
pub use relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};

pub use output_constraint::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, SumOfProductsVisitor, ValueSource,
};

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::opening_proof::OpeningId;
use crate::utils::math::Math;

/// Errors that can occur during BlindFold verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    SpartanSumcheckFailed(usize),
    WrongSpartanProofLength { expected: usize, got: usize },
    DegreeBoundExceeded { expected: usize, got: usize },
    MalformedProof,
    EOpeningFailed,
    OuterClaimMismatch,
    WrongInnerSumcheckLength { expected: usize, got: usize },
    InnerSumcheckFailed(usize),
    WOpeningFailed,
    FinalClaimMismatch,
    EvalCommitmentMismatch,
}

/// ZK data collected during `prove_zk` for a single batched sumcheck stage.
///
/// Stores polynomial coefficients, blinding factors, and Pedersen commitments
/// needed to construct the BlindFold witness.
#[derive(Clone, Debug)]
pub struct ZkStageData<F: JoltField, C: JoltCurve<F = F>> {
    pub initial_claim: F,
    pub round_commitments: Vec<C::G1>,
    pub poly_coeffs: Vec<Vec<F>>,
    pub blinding_factors: Vec<F>,
    pub challenges: Vec<F::Challenge>,
    pub batching_coefficients: Vec<F>,
    pub output_constraints: Vec<Option<OutputClaimConstraint>>,
    pub constraint_challenge_values: Vec<Vec<F>>,
    pub input_constraints: Vec<InputClaimConstraint>,
    pub input_constraint_challenge_values: Vec<Vec<F>>,
    pub input_claim_scaling_exponents: Vec<usize>,
    pub output_claims: Vec<(OpeningId, F)>,
    pub output_claims_blindings: Vec<F>,
    pub output_claims_commitments: Vec<C::G1>,
}

/// ZK data for a uni-skip first round.
#[derive(Clone, Debug)]
pub struct UniSkipStageData<F: JoltField, C: JoltCurve<F = F>> {
    pub input_claim: F,
    pub poly_coeffs: Vec<F>,
    pub blinding_factor: F,
    pub challenge: F::Challenge,
    pub poly_degree: usize,
    pub commitment: C::G1,
    pub input_constraint: InputClaimConstraint,
    pub input_constraint_challenge_values: Vec<F>,
    pub output_constraint: Option<OutputClaimConstraint>,
    pub output_constraint_challenge_values: Vec<F>,
    pub output_claims: Vec<(OpeningId, F)>,
    pub output_claims_blindings: Vec<F>,
    pub output_claims_commitments: Vec<C::G1>,
}

/// ZK data from the PCS opening proof stage.
#[derive(Clone, Debug)]
pub struct OpeningProofData<F: JoltField> {
    pub opening_ids: Vec<OpeningId>,
    pub constraint_coeffs: Vec<F>,
    pub joint_claim: F,
    pub y_blinding: F,
}

/// Accumulates BlindFold-specific data during ZK proving.
#[derive(Clone, Debug)]
pub struct BlindFoldAccumulator<F: JoltField, C: JoltCurve<F = F>> {
    stage_data: Vec<ZkStageData<F, C>>,
    uniskip_data: Vec<UniSkipStageData<F, C>>,
    opening_proof_data: Option<OpeningProofData<F>>,
}

impl<F: JoltField, C: JoltCurve<F = F>> BlindFoldAccumulator<F, C> {
    pub fn new() -> Self {
        Self {
            stage_data: Vec::new(),
            uniskip_data: Vec::new(),
            opening_proof_data: None,
        }
    }

    pub fn push_stage_data(&mut self, data: ZkStageData<F, C>) {
        self.stage_data.push(data);
    }

    pub fn take_stage_data(&mut self) -> Vec<ZkStageData<F, C>> {
        std::mem::take(&mut self.stage_data)
    }

    pub fn push_uniskip_data(&mut self, data: UniSkipStageData<F, C>) {
        self.uniskip_data.push(data);
    }

    pub fn take_uniskip_data(&mut self) -> Vec<UniSkipStageData<F, C>> {
        std::mem::take(&mut self.uniskip_data)
    }

    pub fn set_opening_proof_data(&mut self, data: OpeningProofData<F>) {
        self.opening_proof_data = Some(data);
    }

    pub fn take_opening_proof_data(&mut self) -> OpeningProofData<F> {
        self.opening_proof_data
            .take()
            .expect("opening_proof_data must be set before prove_blindfold")
    }
}

impl<F: JoltField, C: JoltCurve<F = F>> Default for BlindFoldAccumulator<F, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Values baked into R1CS matrix coefficients instead of being public inputs in Z.
///
/// Both prover and verifier construct the same `BakedPublicInputs` from Fiat-Shamir
/// transcript values, then pass them to the R1CS builder.
#[derive(Clone, Debug, Default)]
pub struct BakedPublicInputs<F> {
    pub challenges: Vec<F>,
    pub initial_claims: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_constraint_challenges: Vec<F>,
    pub input_constraint_challenges: Vec<F>,
    pub extra_constraint_challenges: Vec<F>,
}

#[cfg(any(test, feature = "test-feature", feature = "zk"))]
impl<F: JoltField> BakedPublicInputs<F> {
    pub fn from_witness(
        witness: &witness::BlindFoldWitness<F>,
        stage_configs: &[StageConfig],
    ) -> Self {
        let mut challenges = Vec::new();
        for stage in &witness.stages {
            for round in &stage.rounds {
                challenges.push(round.challenge);
            }
        }

        let mut batching_coefficients = Vec::new();
        let mut output_constraint_challenges = Vec::new();
        let mut input_constraint_challenges = Vec::new();

        for (stage_idx, stage) in witness.stages.iter().enumerate() {
            let config = &stage_configs[stage_idx];

            if config.initial_input.is_some() {
                if let Some(witness::FinalOutputWitness::General {
                    challenge_values, ..
                }) = &stage.initial_input
                {
                    input_constraint_challenges.extend_from_slice(challenge_values);
                }
            }

            if let Some(ref fout) = config.final_output {
                if fout.constraint.is_some() {
                    if let Some(witness::FinalOutputWitness::General {
                        challenge_values, ..
                    }) = &stage.final_output
                    {
                        output_constraint_challenges.extend_from_slice(challenge_values);
                    }
                } else if let Some(witness::FinalOutputWitness::Linear {
                    batching_coefficients: coeffs,
                    ..
                }) = &stage.final_output
                {
                    batching_coefficients.extend_from_slice(coeffs);
                }
            }
        }

        let mut extra_constraint_challenges = Vec::new();
        for ew in &witness.extra_constraints {
            extra_constraint_challenges.extend_from_slice(&ew.challenge_values);
        }

        BakedPublicInputs {
            challenges,
            initial_claims: witness.initial_claims.clone(),
            batching_coefficients,
            output_constraint_challenges,
            input_constraint_challenges,
            extra_constraint_challenges,
        }
    }
}

/// Grid layout parameters for Hyrax-style openings.
///
/// W is laid out as an R' x C grid:
/// - Rows 0..total_rounds: coefficient rows (one per sumcheck round)
/// - Rows R_coeff..R_coeff+noncoeff_rows: non-coefficient values (packed)
/// - Remaining rows: zero padding to R'
#[derive(Clone, Debug)]
pub struct HyraxParams {
    pub C: usize,
    pub R_coeff: usize,
    pub R_prime: usize,
    pub noncoeff_count: usize,
    pub total_rounds: usize,
    pub output_claims_rows: usize,
}

impl HyraxParams {
    pub fn e_grid(&self, num_constraints: usize) -> (usize, usize) {
        Self::e_grid_static(self.C, num_constraints)
    }

    pub fn e_grid_static(hyrax_C: usize, num_constraints: usize) -> (usize, usize) {
        let padded_e_len = num_constraints.next_power_of_two();
        let C_E = hyrax_C.min(padded_e_len);
        let R_E = padded_e_len / C_E;
        (R_E, C_E)
    }

    pub fn log_R_prime(&self) -> usize {
        self.R_prime.log_2()
    }

    pub fn log_C(&self) -> usize {
        self.C.log_2()
    }

    pub fn regular_noncoeff_rows(&self) -> usize {
        self.noncoeff_count.div_ceil(self.C)
    }

    pub fn total_noncoeff_rows(&self) -> usize {
        self.output_claims_rows + self.regular_noncoeff_rows()
    }
}

/// Compute Hyrax grid parameters from stage configs.
pub fn compute_hyrax_params(
    stage_configs: &[StageConfig],
    noncoeff_count: usize,
    output_claims_rows: usize,
) -> HyraxParams {
    let total_rounds: usize = stage_configs.iter().map(|s| s.num_rounds).sum();
    let max_coeffs = stage_configs
        .iter()
        .map(|c| c.poly_degree + 1)
        .max()
        .unwrap_or(1);
    let C = max_coeffs.next_power_of_two();
    let R_coeff = if total_rounds == 0 {
        1
    } else {
        total_rounds.next_power_of_two()
    };
    let regular_noncoeff_rows = noncoeff_count.div_ceil(C);
    let R_prime = (R_coeff + output_claims_rows + regular_noncoeff_rows).next_power_of_two();

    HyraxParams {
        C,
        R_coeff,
        R_prime,
        noncoeff_count,
        total_rounds,
        output_claims_rows,
    }
}

/// Configuration for final output binding at end of a chain.
#[derive(Clone, Debug, Default)]
pub struct ClaimBindingConfig {
    pub num_evaluations: usize,
    pub constraint: Option<OutputClaimConstraint>,
    pub exact_num_witness_vars: Option<usize>,
}

impl ClaimBindingConfig {
    pub fn new(num_evaluations: usize) -> Self {
        Self {
            num_evaluations,
            constraint: None,
            exact_num_witness_vars: None,
        }
    }

    pub fn with_constraint(constraint: OutputClaimConstraint) -> Self {
        let num_evaluations = constraint.required_openings.len();
        Self {
            num_evaluations,
            constraint: Some(constraint),
            exact_num_witness_vars: None,
        }
    }

    pub fn verifier_placeholder(num_witness_vars: usize) -> Self {
        Self {
            num_evaluations: 0,
            constraint: None,
            exact_num_witness_vars: Some(num_witness_vars),
        }
    }
}

/// Configuration for a single sumcheck stage.
#[derive(Clone, Debug)]
pub struct StageConfig {
    pub num_rounds: usize,
    pub poly_degree: usize,
    pub starts_new_chain: bool,
    pub uniskip_power_sums: Option<Vec<i128>>,
    pub final_output: Option<ClaimBindingConfig>,
    pub initial_input: Option<ClaimBindingConfig>,
}

impl StageConfig {
    pub fn new(num_rounds: usize, poly_degree: usize) -> Self {
        Self {
            num_rounds,
            poly_degree,
            starts_new_chain: false,
            uniskip_power_sums: None,
            final_output: None,
            initial_input: None,
        }
    }

    pub fn new_chain(num_rounds: usize, poly_degree: usize) -> Self {
        Self {
            num_rounds,
            poly_degree,
            starts_new_chain: true,
            uniskip_power_sums: None,
            final_output: None,
            initial_input: None,
        }
    }

    pub fn new_uniskip(poly_degree: usize, power_sums: Vec<i128>) -> Self {
        Self {
            num_rounds: 1,
            poly_degree,
            starts_new_chain: false,
            uniskip_power_sums: Some(power_sums),
            final_output: None,
            initial_input: None,
        }
    }

    pub fn new_uniskip_chain(poly_degree: usize, power_sums: Vec<i128>) -> Self {
        Self {
            num_rounds: 1,
            poly_degree,
            starts_new_chain: true,
            uniskip_power_sums: Some(power_sums),
            final_output: None,
            initial_input: None,
        }
    }

    pub fn with_final_output(mut self, num_evaluations: usize) -> Self {
        self.final_output = Some(ClaimBindingConfig::new(num_evaluations));
        self
    }

    pub fn with_constraint(mut self, constraint: OutputClaimConstraint) -> Self {
        self.final_output = Some(ClaimBindingConfig::with_constraint(constraint));
        self
    }

    pub fn with_input_constraint(mut self, constraint: InputClaimConstraint) -> Self {
        self.initial_input = Some(ClaimBindingConfig::with_constraint(constraint));
        self
    }

    pub fn is_uniskip(&self) -> bool {
        self.uniskip_power_sums.is_some()
    }

    pub fn has_initial_claim_var(&self) -> bool {
        self.initial_input
            .as_ref()
            .and_then(|ii| ii.constraint.as_ref())
            .is_some_and(|c| !c.terms.is_empty())
    }
}

/// Variable index in the witness vector Z.
///
/// For relaxed R1CS, Z = [u, witness...].
/// Index 0 is the scalar u (u=1 for non-relaxed, u=u1+r*u2 for folded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub usize);

impl Variable {
    pub const U: Variable = Variable(0);

    pub const fn new(idx: usize) -> Self {
        Variable(idx)
    }

    pub const fn index(&self) -> usize {
        self.0
    }
}

/// A term in a linear combination: coefficient * variable.
#[derive(Clone, Copy, Debug)]
pub struct Term<F> {
    pub var: Variable,
    pub coeff: F,
}

impl<F: JoltField> Term<F> {
    pub fn new(var: Variable, coeff: F) -> Self {
        Self { var, coeff }
    }

    pub fn one(var: Variable) -> Self {
        Self::new(var, F::one())
    }

    pub fn neg_one(var: Variable) -> Self {
        Self::new(var, -F::one())
    }
}

/// A linear combination of variables: Sum coeff_i * var_i.
#[derive(Clone, Debug, Default)]
pub struct LinearCombination<F> {
    pub terms: Vec<Term<F>>,
}

impl<F: JoltField> LinearCombination<F> {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Create `value * u` where u is the relaxation scalar.
    pub fn constant(value: F) -> Self {
        Self {
            terms: vec![Term::new(Variable::U, value)],
        }
    }

    pub fn variable(var: Variable) -> Self {
        Self {
            terms: vec![Term::one(var)],
        }
    }

    pub fn add_term(mut self, var: Variable, coeff: F) -> Self {
        self.terms.push(Term::new(var, coeff));
        self
    }

    pub fn add_var(mut self, var: Variable) -> Self {
        self.terms.push(Term::one(var));
        self
    }

    pub fn sub_var(mut self, var: Variable) -> Self {
        self.terms.push(Term::neg_one(var));
        self
    }

    pub fn evaluate(&self, z: &[F]) -> F {
        self.terms
            .iter()
            .map(|term| term.coeff * z[term.var.index()])
            .sum()
    }
}

/// An R1CS constraint: (A * Z) * (B * Z) = (C * Z).
#[derive(Clone, Debug)]
pub struct Constraint<F> {
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

impl<F: JoltField> Constraint<F> {
    pub fn new(a: LinearCombination<F>, b: LinearCombination<F>, c: LinearCombination<F>) -> Self {
        Self { a, b, c }
    }

    pub fn is_satisfied(&self, z: &[F]) -> bool {
        let a_val = self.a.evaluate(z);
        let b_val = self.b.evaluate(z);
        let c_val = self.c.evaluate(z);
        a_val * b_val == c_val
    }
}

/// Pedersen generator count needed for BlindFold R1CS.
pub fn pedersen_generator_count_for_r1cs<F: JoltField>(hyrax: &HyraxParams) -> usize {
    hyrax.C
}
