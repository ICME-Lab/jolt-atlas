use std::{cell::RefCell, rc::Rc};

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{BIG_ENDIAN, OpeningPoint},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::{AppendToTranscript, Transcript},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::{BatchedSumcheck, SumcheckInstance},
    witness::VirtualPolynomial,
};

/// Implements the sumcheck prover for the core Shout PIOP when d = 1. See
/// Figure 5 from the Twist+Shout paper.
/// Proves that rv = sum(ra * val) by proving equality at given r_cycle.
/// Might use in our case by committing to rv and ra, then open rv(r_cycle) and prove it equals sum( ra(r_cycle, k) * val(k) )
/// the above, with aditionnal boolean and hamming weight check, should prove that ra is one-hot encoding for rv.
/// Now two choices: or I try to modify it so that we can prove a given C is gather(rv, val).
/// PS: Need to make sure ra encodes the right value with a raf-eval: raf(r_cycle) = sum( ra(r_cycle, k) * Int(k) )
///
/// I think of an additional variable in val and rv, let n the number of words for each dictionnary entry
/// prove for all n, rv(r_cycle, n) = sum( ra(r_cycle, k) * val(k, n) ).
/// Each entry in raf maps to n words of dictionnary
///
/// Allegories:
/// raf: vector of indexes to gather
/// ra: one-hot encoding of raf
/// val: dictionnary of words for each index
/// rv: output of gather for index in raf
pub struct ShoutProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ShoutProof::prove")]
    pub fn prove<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, ProofTranscript, PCS>,
        lookup_table: Vec<F>,
        read_addresses: Vec<usize>,
        word_dims: usize,
    ) -> Self {
        let K = lookup_table.len();
        let T = read_addresses.len();

        // Recover challenge built accross previous rounds
        let (r_c, _) = sm
            .get_prover_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            );

        let r_x = r_c.split_at(T.log_2()).0.r;

        let E: Vec<F> = EqPolynomial::evals(&r_x);
        let F: Vec<F> = (0..K)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum()
            })
            .collect();

        let mut core_piop_sumcheck =
            ShoutSumcheck::new_prover(sm, lookup_table.clone(), read_addresses.clone());

        let mut booleanity_sumcheck =
            BooleanitySumcheck::new_prover(sm, read_addresses, F, word_dims);

        let accumulator = sm.get_prover_accumulator();
        let (sumcheck_proof, _r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut core_piop_sumcheck, &mut booleanity_sumcheck],
            Some(accumulator),
            &mut *sm.get_transcript().borrow_mut(),
        );

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Self { sumcheck_proof }
    }

    pub fn verify<'a, PCS: CommitmentScheme<Field = F>>(
        &self,
        sm: &mut StateManager<'a, F, ProofTranscript, PCS>,
        K: usize,
        T: usize,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let core_piop_sumcheck = ShoutSumcheck::new_verifier(sm, K, T);

        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(sm, T.log_2(), K.log_2());

        let accumulator = sm.get_verifier_accumulator();
        let r_sumcheck = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![&core_piop_sumcheck, &booleanity_sumcheck],
            Some(accumulator),
            &mut *sm.get_transcript().borrow_mut(),
        )?;

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Ok(r_sumcheck)
    }
}

struct ShoutProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
}

impl<F: JoltField> ShoutProverState<F> {
    fn initialize(
        // <'a, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
        // TODO(AntoineF4C5): Use prover state to recover values for a/Ra and B
        // state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        lookup_table: Vec<F>,
        F: Vec<F>,
    ) -> Self {
        assert_eq!(F.iter().sum::<F>(), F::one());

        let ra = MultilinearPolynomial::from(F.clone());
        let val = MultilinearPolynomial::from(lookup_table);

        Self { ra, val }
    }
}

struct ShoutSumcheck<F: JoltField> {
    prover_state: Option<ShoutProverState<F>>,
    rv_claim_c: F,
    K: usize,
    // random challenge to batch rv and Hamming Weight sumchecks
    z: F,
    // Index of the gather instance
    index: usize,
    r_x: Vec<F>,
    r_y: Vec<F>,
}

impl<F: JoltField> ShoutSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        lookup_table: Vec<F>,
        read_addresses: Vec<usize>,
    ) -> Self {
        let T = read_addresses.len();
        let K = lookup_table.len();

        let (r_c, rv_claim) = sm
            .get_prover_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            );
        let (r_x, r_y) = r_c.r.split_at(T.log_2());
        let E: Vec<F> = EqPolynomial::evals(r_x);
        let F: Vec<F> = (0..K)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum()
            })
            .collect();

        let core_piop_prover_state = ShoutProverState::initialize(lookup_table.clone(), F);

        // Random challenge used to batch rv-check with hamming-weight check
        let z = sm.get_transcript().borrow_mut().challenge_scalar();

        ShoutSumcheck {
            prover_state: Some(core_piop_prover_state),
            // TODO(AntoineF4C5): populate
            index: 0,
            rv_claim_c: rv_claim,
            K,
            z,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        K: usize,
        T: usize,
    ) -> Self {
        let (r_c, rv_claim) = sm
            .get_verifier_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            );
        let (r_x, r_y) = r_c.r.split_at(T.log_2());

        // Random challenge used to batch rv-check with hamming-weight check
        let z = sm.get_transcript().borrow_mut().challenge_scalar();

        Self {
            prover_state: None,
            rv_claim_c: rv_claim,
            K,
            z,
            index: 0,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ShoutSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
        self.rv_claim_c + self.z
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ShoutProverState { ra, val, .. } = self.prover_state.as_ref().unwrap();

        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (self.z + val_evals[0]),
                    ra_evals[1] * (self.z + val_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let ShoutProverState { ra, val, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self.prover_state.as_ref().unwrap();

        let ra_claim = prover_state.ra.final_sumcheck_claim();

        let val_claim = prover_state.val.final_sumcheck_claim();

        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
            ra_claim,
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
            val_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        // This check should be the final claim of sumcheck: C(r_x, r_y) = Sum_k( ra(r_x, k) * (b(k, r_y) + z) )
        // This proves that the output of this instruction is indeed equal to the matrix-multiplication of ra and b,
        // where ra is claimed to be the one-hot encoding of input a.
        // This sumcheck also proves (Hamming Weight) that the sum of all entries at all rows of ra equals 1 (part of proving one-hot encoding).
        // It is also required to prove (Booleanity) that each value in ra is in the set {0, 1}. This and Hamming Weight assert that ra is a one-hot encoded vector.
        // We also need to prove (raf-evaluation) that for each row of ra, the index of the unique 1 equals the value of corresponding entry in input a. (a(r_x) = Sum_k( ra(r_x, k) * Id(k) )
        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
        );
        let (_, b_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
        );

        ra_claim * (self.z + b_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }
}

struct BooleanityProverState<F: JoltField> {
    read_addresses: Vec<usize>,
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: Vec<F>,
    H: Option<MultilinearPolynomial<F>>,
    F: Vec<F>,
    eq_r_r: F,
}

pub struct BooleanitySumcheck<F: JoltField> {
    log_T: usize,
    log_K: usize,
    prover_state: Option<BooleanityProverState<F>>,
    r_cycle: Vec<F>,
    r_address: Vec<F>,
    index: usize,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        G: Vec<F>,
        one_hot_length: usize,
    ) -> Self {
        let log_K = one_hot_length.log_2();
        let log_T = read_addresses.len().log_2();
        // Generate a random challenge to complete with r_x for RA booleanity sumcheck:
        // r_x spans over the input length (column-length of RA matrix),
        // this random challenge will span over the number of words (row-length of RA matrix)
        let r_words = sm.get_transcript().borrow_mut().challenge_vector(log_K);

        let r_c = sm
            .get_prover_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            )
            .0;
        let r_x = r_c.r.split_at(log_T).0;

        Self {
            prover_state: Some(BooleanityProverState::new(
                read_addresses,
                EqPolynomial::evals(&r_x),
                G,
                &r_words,
                log_K,
            )),
            index: 0,
            log_T,
            log_K,
            r_cycle: r_x.to_vec(),
            r_address: r_words,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        log_T: usize,
        log_K: usize,
    ) -> Self {
        let r_words = sm.get_transcript().borrow_mut().challenge_vector(log_K);

        let r_c = sm
            .get_verifier_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            )
            .0;
        let r_x = r_c.r.split_at(log_T).0;

        Self {
            prover_state: None,
            index: 0,
            log_T,
            log_K,
            r_cycle: r_x.to_vec(),
            r_address: r_words,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        read_addresses: Vec<usize>,
        eq_r_cycle: Vec<F>,
        G: Vec<F>,
        r_address: &[F],
        log_K: usize,
    ) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_address));

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F_vec[0] = F::one();

        let D = MultilinearPolynomial::from(eq_r_cycle);

        BooleanityProverState {
            read_addresses,
            B,
            D,
            H: None,
            G,
            F: F_vec,
            eq_r_r: F::zero(),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.log_K {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < self.log_K {
            // Phase 1: Bind B and update F
            ps.B.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = ps.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H
            if round == self.log_K - 1 {
                let mut read_addresses = std::mem::take(&mut ps.read_addresses);
                let f_ref = &ps.F;
                ps.H = Some({
                    let coeffs: Vec<F> = std::mem::take(&mut read_addresses)
                        .into_par_iter()
                        .map(|j| f_ref[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                });
                ps.eq_r_r = ps.B.final_sumcheck_claim();

                // Drop G arrays, F array, and read_addresses as they're no longer needed in phase 2
                let g = std::mem::take(&mut ps.G);
                drop_in_background_thread(g);

                let f = std::mem::take(&mut ps.F);
                drop_in_background_thread(f);

                drop_in_background_thread(read_addresses);
            }
        } else {
            let H = ps.H.as_mut().unwrap();
            rayon::join(
                || H.bind_parallel(r_j, BindingOrder::LowToHigh),
                || ps.D.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
        );

        EqPolynomial::mle(
            r,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.r_cycle.iter().cloned().rev())
                .collect::<Vec<F>>(),
        ) * (ra_claim.square() - ra_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point[..self.log_K].reverse();
        opening_point[self.log_K..].reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();
        let ra_claim = ps.H.as_ref().unwrap().final_sumcheck_claim();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
            opening_point,
            ra_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::PrecompileExecution,
            opening_point,
        );
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
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

        let univariate_poly_evals: [F; 3] = (0..p.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals =
                    p.B.sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = p.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F = p.G[k_G] * F_k;
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
        let p = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..p.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals =
                    p.D.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H = p.H.as_ref().unwrap();
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
            p.eq_r_r * univariate_poly_evals[0],
            p.eq_r_r * univariate_poly_evals[1],
            p.eq_r_r * univariate_poly_evals[2],
        ]
    }
}

pub fn prove_core_shout_piop<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());

    // Sumcheck for the core Shout PIOP (Figure 5)
    let num_rounds = K.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    // One-hot encoding of read-adresses, folded over cycle variables
    let F: Vec<F> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let sumcheck_claim: F = F
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();
    let mut previous_claim = sumcheck_claim;

    let mut ra = MultilinearPolynomial::from(F);
    let mut val = MultilinearPolynomial::from(lookup_table);

    const DEGREE: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _round in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the Booleanity check in step 3 of
/// Figure 6 in the Twist+Shout paper. The efficient implementation of this
/// sumcheck is described in Section 6.3.
pub fn prove_booleanity<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F) {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = read_addresses.len();
    debug_assert_eq!(D.len(), T);
    debug_assert_eq!(G.len(), K);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(r)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
    F[0] = F::one();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c: [[F; DEGREE]; 2] = [
        [
            F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
            F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
            F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
        ],
        [
            F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
            F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
            F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
        ],
    ];

    // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    for round in 0..K.log_2() {
        let m = round + 1;

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))

                        [
                            G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                            G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                            G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
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

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind_parallel(r_j, BindingOrder::LowToHigh);

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << round);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    let eq_r_r = B.final_sumcheck_claim();
    let H: Vec<F> = read_addresses.par_iter().map(|&k| F[k]).collect();
    let mut H = MultilinearPolynomial::from(H);
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());

    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
        #[cfg(test)]
        {
            let expected: F = eq_r_r
                * (0..H.len())
                    .map(|j| {
                        let D_j = D.get_bound_coeff(j);
                        let H_j = H.get_bound_coeff(j);
                        D_j * (H_j.square() - H_j)
                    })
                    .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                    D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                    D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
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

        univariate_poly_evals = [
            eq_r_r * univariate_poly_evals[0],
            eq_r_r * univariate_poly_evals[1],
            eq_r_r * univariate_poly_evals[2],
        ];

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind_parallel(r_j, BindingOrder::LowToHigh),
            || H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = H.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        r_cycle,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the Hamming weight 1 check in step 5 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle_prime: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle_prime.len());

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle_prime);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let mut ra = MultilinearPolynomial::from(F);
    let mut previous_claim = F::one();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_eval: F = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| ra.get_bound_coeff(2 * i))
            .sum();

        let univariate_poly =
            UniPoly::from_evals(&[univariate_poly_eval, previous_claim - univariate_poly_eval]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_double_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the raf-evaluation sumcheck in step 6 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_raf_evaluation<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle.len());

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let mut ra = MultilinearPolynomial::from(F);
    let mut int = IdentityPolynomial::new(num_rounds);

    let mut previous_claim = claimed_evaluation;

    const DEGREE: usize = 2;

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let int_evals: Vec<F> = int.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * int_evals[0], ra_evals[1] * int_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || int.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    (SumcheckInstanceProof::new(compressed_polys), ra_claim)
}

#[cfg(test)]
mod tests {
    use crate::jolt::{
        JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
        bytecode::BytecodePreprocessing, precompiles::PrecompilePreprocessing,
        sumcheck::SingleSumcheck, trace::JoltONNXCycle,
    };

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero, test_rng};
    use itertools::Itertools;
    use jolt_core::{poly::commitment::mock::MockCommitScheme, transcripts::Blake2bTranscript};
    use onnx_tracer::{ProgramIO, tensor::Tensor};
    use rand_core::RngCore;

    #[test]
    #[ignore] // Doesn't work yet: core_piop ?
    fn shout_e2e() {
        // Number of lines that we want to gather from the dictionnary
        const NUM_INPUTS: usize = 16;
        // The dimension of a dictionnary line
        const WORD_DIMS: usize = 8;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 32;

        let mut rng = test_rng();

        // Random a vector a, with values in 0..K, index of different possible words in the dictionnary
        let read_indexes: Vec<usize> = (0..NUM_INPUTS)
            .map(|_| rng.next_u32() as usize % NUM_WORDS)
            .collect();

        // Gen A, the one-hot encoding of vector a.
        // TODO(AntoineF4C5): Prover should commit to this tensor, this is for now read as a virtual polynomial, i.e VirtualPolynomial::GatherRa(usize) variant
        let _A: Vec<usize> = read_indexes
            .iter()
            .flat_map(|&address| {
                let mut line = vec![0; NUM_WORDS];
                line[address] = 1;
                line
            })
            .collect();

        // Dictionnary; gather's matrix from which desired inputs are gathered
        let B: Vec<Fr> = (0..NUM_WORDS * WORD_DIMS)
            .map(|_| Fr::random(&mut rng))
            .collect();

        let bytecode_pp = BytecodePreprocessing::default();
        // let precompile_pp = PrecompilePreprocessing::preprocess(&bytecode_pp);
        let shared_pp = JoltSharedPreprocessing {
            bytecode: bytecode_pp,
            precompiles: PrecompilePreprocessing::empty(),
            fp_lookups: Default::default(),
        };

        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_pp.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_pp,
            };
        let program_io = ProgramIO {
            input: Tensor::new(None, &[]).unwrap(),
            output: Tensor::new(None, &[]).unwrap(),
            min_lookup_input: 0,
            max_lookup_input: 0,
        };

        let trace = vec![JoltONNXCycle::no_op(); 32];
        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            trace.clone(),
            program_io.clone(),
        );

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Expected output of the gather node: a matrix where each row `i` corresponds to the row at index `a[i]` in B
        let output: Vec<Fr> = read_indexes
            .iter()
            .flat_map(|index| B[index * WORD_DIMS..(index + 1) * WORD_DIMS].to_vec())
            .collect();
        let prover_transcript = prover_sm.get_transcript();
        let mut prover_transcript_ref = prover_transcript.borrow_mut();
        // Reduces to a single evaluation over dimension of `read_indexes`
        let r_x: Vec<Fr> = prover_transcript_ref.challenge_vector(NUM_INPUTS.log_2());
        // Reduces over a single evaluation over a dictionnary word's embedding size
        let r_y: Vec<Fr> = prover_transcript_ref.challenge_vector(WORD_DIMS.log_2());
        drop(prover_transcript_ref);

        // Insert claim from previous stage of jolt-dag
        let r_c = [r_x.clone(), r_y.clone()].concat();
        let rv_claim = MultilinearPolynomial::from(output).evaluate(&r_c);
        prover_sm
            .get_prover_accumulator()
            .borrow_mut()
            .append_virtual(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
                r_c.clone().into(),
                rv_claim,
            );
        //-------------------------------------------------------------------------------------------------

        let eq_r_x = EqPolynomial::evals(&r_x);
        let eq_r_y = EqPolynomial::evals(&r_y);

        let mut ra_folded = vec![Fr::zero(); NUM_WORDS];
        read_indexes.iter().enumerate().for_each(|(i, &address)| {
            ra_folded[address] += eq_r_x[i];
        });

        let B_folded: Vec<Fr> = B
            .chunks(WORD_DIMS)
            .map(|B_chunk| {
                B_chunk
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(b, e)| b * e)
                    .sum::<Fr>()
            })
            .collect();

        let shout_proof =
            ShoutProof::prove(&mut prover_sm, B_folded.clone(), read_indexes, WORD_DIMS);

        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            1 << 8,
            prover_sm.twist_sumcheck_switch_index,
        );

        // Take claims
        let prover_acc = prover_sm.get_prover_accumulator();
        let prover_acc_borrow = prover_acc.borrow();
        let verifier_accumulator = verifier_sm.get_verifier_accumulator();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_acc_borrow
                .openings_mut()
                .insert(*key, (empty_point, *value));
        }
        drop((prover_acc_borrow, verifier_acc_borrow));

        //--------------------- Simulate challenges from previous stages of jolt-dag ---------------------
        let verifier_transcript = verifier_sm.get_transcript();
        let mut verifier_transcript_ref = verifier_transcript.borrow_mut();
        let _r_x: Vec<Fr> = verifier_transcript_ref.challenge_vector(NUM_INPUTS.log_2());
        let _r_y: Vec<Fr> = verifier_transcript_ref.challenge_vector(WORD_DIMS.log_2());
        assert_eq!((r_x, r_y), (_r_x, _r_y));
        drop(verifier_transcript_ref);

        // Insert opening point for the simulated claim from previous jolt-dag stage
        verifier_sm
            .get_verifier_accumulator()
            .borrow_mut()
            .append_virtual(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
                r_c.into(),
            );
        //-------------------------------------------------------------------------------------------------

        let verification_result = shout_proof.verify(&mut verifier_sm, NUM_WORDS, NUM_INPUTS);

        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn core_shout_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let (sumcheck_proof, _, sumcheck_claim, _) =
            prove_core_shout_piop(lookup_table, read_addresses, &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            TABLE_SIZE.log_2(),
            2,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn booleanity_sumcheck() {
        let bytecode_pp = BytecodePreprocessing::default();
        // let precompile_pp = PrecompilePreprocessing::preprocess(&bytecode_pp);
        let shared_pp = JoltSharedPreprocessing {
            bytecode: bytecode_pp,
            precompiles: PrecompilePreprocessing::empty(),
            fp_lookups: Default::default(),
        };

        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_pp.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_pp,
            };
        let program_io = ProgramIO {
            input: Tensor::new(None, &[]).unwrap(),
            output: Tensor::new(None, &[]).unwrap(),
            min_lookup_input: 0,
            max_lookup_input: 0,
        };

        let trace = vec![JoltONNXCycle::no_op(); 32];

        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            trace.clone(),
            program_io.clone(),
        );

        const NUM_WORDS: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of dimensions per word
        const WORDS_DIM: usize = 1 << 3;

        let mut rng = test_rng();

        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % NUM_WORDS)
            .collect();
        let dummy_table = vec![Fr::zero(); NUM_WORDS * WORDS_DIM];

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Expected output of the gather node: a matrix where each row `i` corresponds to the row at index `a[i]` in B
        let output: Vec<Fr> = read_addresses
            .iter()
            .flat_map(|&index| dummy_table[index * WORDS_DIM..(index + 1) * WORDS_DIM].to_vec())
            .collect();
        let prover_transcript = prover_sm.get_transcript();
        let mut prover_transcript_ref = prover_transcript.borrow_mut();
        // Reduces to a single evaluation over dimension of `read_indexes`
        let r_x: Vec<Fr> = prover_transcript_ref.challenge_vector(NUM_LOOKUPS.log_2());
        // // Reduces over a single evaluation over a dictionnary word's embedding size
        let r_y: Vec<Fr> = prover_transcript_ref.challenge_vector(WORDS_DIM.log_2());
        drop(prover_transcript_ref);

        // Insert claim from previous stage of jolt-dag
        let r_c = [r_x.clone(), r_y.clone()].concat();
        let rv_claim = MultilinearPolynomial::from(output).evaluate(&r_c);
        prover_sm
            .get_prover_accumulator()
            .borrow_mut()
            .append_virtual(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
                r_c.clone().into(),
                rv_claim,
            );
        //-------------------------------------------------------------------------------------------------

        let E: Vec<Fr> = EqPolynomial::evals(&r_x);
        let F: Vec<Fr> = (0..NUM_WORDS)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum()
            })
            .collect();

        let mut booleanity_sumcheck =
            BooleanitySumcheck::new_prover(&mut prover_sm, read_addresses, F, NUM_WORDS);

        let (proof, _r_sumcheck) = SingleSumcheck::prove(
            &mut booleanity_sumcheck,
            Some(prover_sm.get_prover_accumulator()),
            &mut *prover_sm.get_transcript().borrow_mut(),
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        // verifier_transcript.compare_to(*prover_transcript.borrow_mut());
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_WORDS.log_2());
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            1 << 8,
            prover_sm.twist_sumcheck_switch_index,
        );

        // Take claims
        let prover_acc = prover_sm.get_prover_accumulator();
        let prover_acc_borrow = prover_acc.borrow();
        let verifier_accumulator = verifier_sm.get_verifier_accumulator();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_acc_borrow
                .openings_mut()
                .insert(*key, (empty_point, *value));
        }
        drop((prover_acc_borrow, verifier_acc_borrow));

        //--------------------- Simulate challenges from previous stages of jolt-dag ---------------------
        let verifier_transcript = verifier_sm.get_transcript();
        let mut verifier_transcript_ref = verifier_transcript.borrow_mut();
        let _r_x: Vec<Fr> = verifier_transcript_ref.challenge_vector(NUM_LOOKUPS.log_2());
        let _r_y: Vec<Fr> = verifier_transcript_ref.challenge_vector(WORDS_DIM.log_2());
        assert_eq!((r_x, r_y), (_r_x, _r_y));
        drop(verifier_transcript_ref);

        // Insert opening point for the simulated claim from previous jolt-dag stage
        verifier_sm
            .get_verifier_accumulator()
            .borrow_mut()
            .append_virtual(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
                r_c.into(),
            );
        //-------------------------------------------------------------------------------------------------

        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(
            &mut verifier_sm,
            NUM_LOOKUPS.log_2(),
            NUM_WORDS.log_2(),
        );

        let res = SingleSumcheck::verify(
            &booleanity_sumcheck,
            &proof,
            Some(verifier_sm.get_verifier_accumulator()),
            &mut *verifier_sm.get_transcript().borrow_mut(),
        );

        assert!(
            res.is_ok(),
            "Booleanity verification failed with error: {:?}",
            res.err()
        );
    }

    #[test]
    fn hamming_weight_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let r_cycle_prime: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let (sumcheck_proof, _, _) = prove_hamming_weight(
            lookup_table,
            read_addresses,
            r_cycle_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(Fr::one(), TABLE_SIZE.log_2(), 1, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn raf_evaluation_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();
        let raf = MultilinearPolynomial::from(
            read_addresses.iter().map(|a| *a as u32).collect::<Vec<_>>(),
        );

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let raf_eval = raf.evaluate(&r_cycle);
        let (sumcheck_proof, _) = prove_raf_evaluation(
            lookup_table,
            read_addresses,
            r_cycle,
            raf_eval,
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(raf_eval, TABLE_SIZE.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
