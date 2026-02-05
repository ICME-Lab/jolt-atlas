use crate::onnx_proof::ops::softmax_axes::softmax::SoftmaxIndex;
use atlas_onnx_tracer::tensor::ops::nonlinearities::{
    SoftmaxTrace, EXP_LUT_SCALE_128, EXP_LUT_SIZE, LOG_EXP_LUT_SIZE,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        math::Math,
        small_scalar::SmallScalar,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;
use std::array;

const READ_RAF_HW_DEGREE_BOUND: usize = 2;

/// Shared prover/verifier parameters for Shout.
#[derive(Clone)]
pub struct ReadRafHwParams<F: JoltField> {
    r_exponentiation_output: Vec<F::Challenge>,
    softmax_index: SoftmaxIndex,
    gamma: F,
    gamma_squared: F,
}

impl<F: JoltField> ReadRafHwParams<F> {
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let r_exponentiation_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_exponentiation_output,
            softmax_index,
            gamma,
            gamma_squared: gamma * gamma,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ReadRafHwParams<F> {
    fn degree(&self) -> usize {
        READ_RAF_HW_DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, read_checking_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        let (_, raf_checking_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        read_checking_claim + self.gamma * raf_checking_claim + self.gamma_squared
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        LOG_EXP_LUT_SIZE
    }
}

pub struct ReadRafHwProver<F: JoltField> {
    params: ReadRafHwParams<F>,
    val: MultilinearPolynomial<F>,
    F: MultilinearPolynomial<F>,
    int: IdentityPolynomial<F>,
}

impl<F: JoltField> ReadRafHwProver<F> {
    pub fn initialize(
        trace: &SoftmaxTrace,
        params: ReadRafHwParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // Add raf claim and implicitly sub claim
        let raf_claim = MultilinearPolynomial::from(trace.abs_centered_logits.to_vec())
            .evaluate(&params.r_exponentiation_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                params.softmax_index.node_idx,
                params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            params.r_exponentiation_output.clone().into(),
            raf_claim,
        );

        let E = EqPolynomial::evals(&params.r_exponentiation_output);
        let lookup_indices = &trace.abs_centered_logits;
        let F = lookup_indices
            .data()
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut local_F, (j, &lookup_index)| {
                    local_F[lookup_index as usize] += E[j];
                    local_F
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut acc, local_F| {
                    for (i, &val) in local_F.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );
        let val = MultilinearPolynomial::from(EXP_LUT_SCALE_128.to_vec());
        Self {
            params,
            val,
            F: MultilinearPolynomial::from(F),
            int: IdentityPolynomial::new(LOG_EXP_LUT_SIZE),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReadRafHwProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self { F, val, int, .. } = self;
        let half_poly_len = val.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let val_evals =
                    val.sumcheck_evals(i, READ_RAF_HW_DEGREE_BOUND, BindingOrder::HighToLow);
                let f_evals =
                    F.sumcheck_evals(i, READ_RAF_HW_DEGREE_BOUND, BindingOrder::HighToLow);
                let int_evals =
                    int.sumcheck_evals(i, READ_RAF_HW_DEGREE_BOUND, BindingOrder::HighToLow);
                [
                    f_evals[0]
                        * (val_evals[0]
                            + self.params.gamma * int_evals[0]
                            + self.params.gamma_squared),
                    f_evals[1]
                        * (val_evals[1]
                            + self.params.gamma * int_evals[1]
                            + self.params.gamma_squared),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.val.bind_parallel(r_j, BindingOrder::HighToLow);
        self.F.bind_parallel(r_j, BindingOrder::HighToLow);
        self.int.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_sparse(
            transcript,
            vec![CommittedPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            )],
            SumcheckId::Execution,
            sumcheck_challenges.to_vec(),
            self.params.r_exponentiation_output.to_vec(),
            vec![self.F.final_sumcheck_claim()],
        );
    }
}

pub struct ReadRafHwVerifier<F: JoltField> {
    params: ReadRafHwParams<F>,
}

impl<F: JoltField> ReadRafHwVerifier<F> {
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ReadRafHwParams::new(softmax_index, accumulator, transcript);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                params.softmax_index.node_idx,
                params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            params.r_exponentiation_output.clone().into(),
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReadRafHwVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxExponentiationRa(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        let val_claim =
            MultilinearPolynomial::from(EXP_LUT_SCALE_128.to_vec()).evaluate(sumcheck_challenges);
        let int_claim = IdentityPolynomial::new(LOG_EXP_LUT_SIZE).evaluate(sumcheck_challenges);
        ra_claim * (val_claim + self.params.gamma * int_claim + self.params.gamma_squared)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = [
            self.params.r_exponentiation_output.as_slice(),
            sumcheck_challenges,
        ]
        .concat();
        accumulator.append_sparse(
            transcript,
            vec![CommittedPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            )],
            SumcheckId::Execution,
            opening_point,
        );
    }
}

const BOOLEANITY_DEGREE_BOUND: usize = 3;

#[derive(Clone)]
pub struct BooleanityParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    r_address: Vec<F::Challenge>,
    softmax_index: SoftmaxIndex,
}

impl<F: JoltField> BooleanityParams<F> {
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_address = transcript.challenge_vector_optimized::<F>(LOG_EXP_LUT_SIZE);
        Self {
            r_node_output,
            r_address,
            softmax_index,
        }
    }

    #[inline]
    pub fn log_K(&self) -> usize {
        LOG_EXP_LUT_SIZE
    }

    #[inline]
    pub fn log_T(&self) -> usize {
        self.r_node_output.len()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanityParams<F> {
    fn degree(&self) -> usize {
        BOOLEANITY_DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.log_K() + self.log_T()
    }
}

pub struct BooleanityProver<F: JoltField> {
    params: BooleanityParams<F>,
    read_addresses: Vec<usize>,
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: Vec<F>,
    H: Option<MultilinearPolynomial<F>>,
    F: Vec<F>,
    eq_r_r: F,
}

impl<F: JoltField> BooleanityProver<F> {
    pub fn initialize(trace: &SoftmaxTrace, params: BooleanityParams<F>) -> Self {
        let lookup_indices = trace
            .abs_centered_logits
            .data()
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>();
        let D = EqPolynomial::evals(&params.r_node_output);
        let B = EqPolynomial::evals(&params.r_address);
        let G = lookup_indices
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut local_G, (j, &lookup_index)| {
                    local_G[lookup_index] += D[j];
                    local_G
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut acc, local_G| {
                    for (i, &val) in local_G.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );
        let mut F: Vec<F> = unsafe_allocate_zero_vec(params.log_K().pow2());
        F[0] = F::one();
        Self {
            params,
            read_addresses: lookup_indices,
            B: MultilinearPolynomial::from(B),
            D: MultilinearPolynomial::from(D),
            G,
            H: None,
            F,
            eq_r_r: F::zero(),
        }
    }

    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let m = round + 1;
        const DEGREE: usize = 3;

        // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        const EQ_KM_C: [[i8; 3]; 2] = [
            [
                1,  // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                -1, // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                -2, // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                0, // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                2, // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                3, // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];

        // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        const EQ_KM_C_SQUARED: [[u8; 3]; 2] = [[1, 1, 4], [0, 4, 9]];

        let univariate_poly_evals: [F; 3] = (0..self.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals = self
                    .B
                    .sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = self.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F = self.G[k_G] * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][0].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][0] as i64)),
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][1].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][1] as i64)),
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][2].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][2] as i64)),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..self.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals = self
                    .D
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H = self.H.as_ref().unwrap();
                let H_evals = H.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let evals = [
                    H_evals[0].square() - H_evals[0],
                    H_evals[1].square() - H_evals[1],
                    H_evals[2].square() - H_evals[2],
                ];
                [
                    D_evals[0] * evals[0],
                    D_evals[1] * evals[1],
                    D_evals[2] * evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        vec![
            self.eq_r_r * univariate_poly_evals[0],
            self.eq_r_r * univariate_poly_evals[1],
            self.eq_r_r * univariate_poly_evals[2],
        ]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BooleanityProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let uni_poly_evals = if round < self.params.log_K() {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        };
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.log_K() {
            // Phase 1: Bind B and update F
            self.B.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = self.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H
            if round == self.params.log_K() - 1 {
                let mut read_addresses = std::mem::take(&mut self.read_addresses);
                let f_ref = &self.F;
                self.H = Some({
                    let coeffs: Vec<F> = std::mem::take(&mut read_addresses)
                        .into_par_iter()
                        .map(|j| f_ref[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                });
                self.eq_r_r = self.B.final_sumcheck_claim();

                // Drop G arrays, F array, and read_addresses as they're no longer needed in phase 2
                let g = std::mem::take(&mut self.G);
                drop_in_background_thread(g);

                let f = std::mem::take(&mut self.F);
                drop_in_background_thread(f);

                drop_in_background_thread(read_addresses);
            }
        } else {
            let H = self.H.as_mut().unwrap();
            rayon::join(
                || H.bind_parallel(r_j, BindingOrder::LowToHigh),
                || self.D.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (r_address, r_node_output) = sumcheck_challenges.split_at(self.params.log_K());
        accumulator.append_sparse(
            transcript,
            vec![CommittedPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            )],
            SumcheckId::Booleanity,
            r_address.to_vec(),
            r_node_output.to_vec(),
            vec![self.H.as_ref().unwrap().final_sumcheck_claim()],
        );
    }
}

pub struct BooleanityVerifier<F: JoltField> {
    params: BooleanityParams<F>,
}

impl<F: JoltField> BooleanityVerifier<F> {
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = BooleanityParams::new(softmax_index, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BooleanityVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxExponentiationRa(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Booleanity,
            )
            .1;
        EqPolynomial::mle(
            sumcheck_challenges,
            &self
                .params
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.params.r_node_output.iter().cloned().rev())
                .collect::<Vec<F::Challenge>>(),
        ) * (ra_claim.square() - ra_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_sparse(
            transcript,
            vec![CommittedPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            )],
            SumcheckId::Booleanity,
            sumcheck_challenges.to_vec(),
        );
    }
}
