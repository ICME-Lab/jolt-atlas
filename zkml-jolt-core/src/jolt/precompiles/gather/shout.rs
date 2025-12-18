use std::{cell::RefCell, rc::Rc};

use itertools::Itertools;
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
    },
    transcripts::Transcript,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

// Implementation of various sumchecks used to prove execution of a gather instruction
//
// RvHwSumcheck proves: C(r_x, r_y) = Sum_k( ra(r_x, k) * (B(k, r_y) + z) )
//  - Read Value ensures that the output of this instruction is indeed equal to the matrix-multiplication of ra and b,
//      where ra is claimed to be the one-hot encoding matrix of input vector A. (row `i` of ra is one-hot encoding of `A[i]`)
//  - Hamming Weight ensures that the sum of all entries at all rows of ra equals 1 (part of proving one-hot encoding).
//
// BooleanitySumcheck proves:  0 = Sum_k,t( eq(r_x, t) * eq(r_y, k) * (ra(t, k)² - ra(t, k)) )
//  - Booleanity ensures that each value in ra is in the set {0, 1}.
//  This and Hamming Weight assert that each row in ra is a one-hot encoded vector.
//
// RafSumcheck proves: a(r_x) = Sum_k( ra(r_x, k) * Id(k) )
//  - Raf Evaluation computes dot product of each row of ra with the identity polynomial.
//  This and Hamming Weight, Booleanity ensures that ra's row `i` is the one-hot encoding of `A[i]`
//
// Putting it in the context of the Gather instructions:
// id   | dims                      | description                                                                                   | name in sumchecks
// A    | [num_lookups]             | first input vector, where `A[i]` is the index of the values to retrieve from the second input | read_addresses
// B    | [num_words * word_dim]    | second input, a matrix where each row holds the values to be retrieved                        | dictionnary
// C    | [num_lookups * word_dim]  | output, a matrix where each row holds the retrieved values from B at index given in A         | output
// ra   | [num_lookups * num_words] | used by all sumchecks, each row holds the one-hot encoding of value held in A                 | ra

pub struct GatherProof {}

impl GatherProof {
    #[tracing::instrument(skip_all, name = "ShoutProof::prove")]
    pub fn gather_prover_instances<
        'a,
        F: JoltField,
        PT: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'a, F, PT, PCS>,
        read_addresses: Vec<usize>,
        dictionnary: Vec<F>,
        // Number of dimension in a word embedding
        word_dim: usize,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let num_lookups = read_addresses.len();
        let num_words = dictionnary.len() / word_dim; // dictionnary is a num_words * word_dim matrix

        // Recover challenge built during previous sumcheck
        let r_c = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            )
            .0;

        let (r_x, r_y) = r_c.split_at(num_lookups.log_2());
        assert_eq!(r_y.len(), word_dim.log_2());

        let E: Vec<F> = EqPolynomial::evals(&r_x.r);
        // ra matrix, evaluated at row r_x
        let F: Vec<F> = (0..num_words)
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

        let eq_r_y = EqPolynomial::evals(&r_y.r);
        // dictionnary, evaluated at column r_y
        let folded_dict: Vec<F> = dictionnary
            .chunks(word_dim)
            .map(|B_chunk| {
                B_chunk
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(&b, &e)| b * e)
                    .sum()
            })
            .collect();
        assert_eq!(folded_dict.len(), num_words);

        let rv_hw_sumcheck =
            RvHwSumcheck::new_prover(sm, read_addresses.clone(), folded_dict, F.clone());

        let booleanity_sumcheck =
            BooleanitySumcheck::new_prover(sm, read_addresses.clone(), F.clone());

        let raf_sumcheck = RafSumcheck::new_prover(sm, read_addresses, F);

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        vec![
            Box::new(rv_hw_sumcheck),
            Box::new(booleanity_sumcheck),
            Box::new(raf_sumcheck),
        ]
    }

    pub fn gather_verifier_instances<
        'a,
        F: JoltField,
        PT: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'a, F, PT, PCS>,
        num_lookups: usize,
        dictionnary_length: usize,
        word_dim: usize,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let num_words = dictionnary_length / word_dim;

        let rv_hw_sumcheck = RvHwSumcheck::new_verifier(sm, num_lookups, num_words);

        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(sm, num_lookups, num_words);

        let raf_sumcheck = RafSumcheck::new_verifier(sm, num_lookups, num_words);

        vec![
            Box::new(rv_hw_sumcheck),
            Box::new(booleanity_sumcheck),
            Box::new(raf_sumcheck),
        ]
    }
}

struct RvHwProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
    dict_folded: MultilinearPolynomial<F>,
}

impl<F: JoltField> RvHwProverState<F> {
    fn initialize(
        // <'a, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
        // TODO(AntoineF4C5): Use prover state to recover values for a/Ra and B
        // state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        dict_folded: Vec<F>,
        F: Vec<F>,
    ) -> Self {
        assert_eq!(F.iter().sum::<F>(), F::one());

        let ra = MultilinearPolynomial::from(F.clone());
        let dict_folded = MultilinearPolynomial::from(dict_folded);

        Self { ra, dict_folded }
    }
}
// RvHwSumcheck proves: C(r_x, r_y) = Sum_k( ra(r_x, k) * (B(k, r_y) + z) )
// r_x and r_y are challenges produced in a sumcheck executed in previous steps
struct RvHwSumcheck<F: JoltField> {
    prover_state: Option<RvHwProverState<F>>,
    // Dimension over which sumcheck is ran
    num_words: usize,
    rv_claim_c: F,
    // random challenge to batch rv and Hamming Weight sumchecks
    z: F,
    r_x: Vec<F>,
    r_y: Vec<F>,
    // Index of the gather instance
    index: usize,
}

impl<F: JoltField> RvHwSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        dictionnary_folded: Vec<F>,
        F: Vec<F>,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = dictionnary_folded.len();

        let (r_c, rv_claim) = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            );
        let (r_x, r_y) = r_c.r.split_at(num_lookups.log_2());

        let rv_hw_prover_state = RvHwProverState::initialize(dictionnary_folded.clone(), F);

        // Random challenge used to batch rv-check with hamming-weight check
        let z = sm.get_transcript().borrow_mut().challenge_scalar();

        Self {
            prover_state: Some(rv_hw_prover_state),
            // TODO(AntoineF4C5): populate
            index: 0,
            rv_claim_c: rv_claim,
            num_words,
            z,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
    ) -> Self {
        let (r_c, rv_claim) = sm
            .get_verifier_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            );
        let (r_x, r_y) = r_c.r.split_at(num_lookups.log_2());

        // Random challenge used to batch rv-check with hamming-weight check
        let z = sm.get_transcript().borrow_mut().challenge_scalar();

        Self {
            prover_state: None,
            num_words,
            rv_claim_c: rv_claim,
            z,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
            index: 0,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RvHwSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
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
        let RvHwProverState { ra, dict_folded } = self.prover_state.as_ref().unwrap();

        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let dict_evals = dict_folded.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (self.z + dict_evals[0]),
                    ra_evals[1] * (self.z + dict_evals[1]),
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
        let RvHwProverState { ra, dict_folded } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || dict_folded.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherRvHw,
        );
        let (_, dict_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::GatherRvHw,
        );

        ra_claim * (self.z + dict_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self.prover_state.as_ref().unwrap();

        let ra_claim = prover_state.ra.final_sumcheck_claim();

        let dict_claim = prover_state.dict_folded.final_sumcheck_claim();

        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherRvHw,
            r_a.into(),
            ra_claim,
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::GatherRvHw,
            r_b.into(),
            dict_claim,
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
            SumcheckId::GatherRvHw,
            r_a.into(),
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::GatherRvHw,
            r_b.into(),
        );
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

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        read_addresses: Vec<usize>,
        eq_r_x: Vec<F>,
        G: Vec<F>,
        r_words: &[F],
        num_words: usize,
    ) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_words));

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(num_words);
        F_vec[0] = F::one();

        let D = MultilinearPolynomial::from(eq_r_x);

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

// BooleanitySumcheck proves:  0 = Sum_k,t( eq(r_x, t) * eq(r_words, k) * (ra(t, k)² - ra(t, k)) )
// r_x is a challenge vector produced in sumchecks executed in previous steps,
// r_words is (for now) a challenge vector produced (for now) at the beginning of the sumcheck
pub struct BooleanitySumcheck<F: JoltField> {
    prover_state: Option<BooleanityProverState<F>>,
    num_lookups: usize,
    num_words: usize,
    r_x: Vec<F>,
    r_words: Vec<F>,
    index: usize,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        G: Vec<F>,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = G.len();

        let r_c = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            )
            .0;
        let r_x = r_c.r.split_at(num_lookups.log_2()).0;

        // Generate a random challenge to complete with r_x for RA booleanity sumcheck:
        // r_x spans over the input length (column-length of RA matrix),
        // this random challenge will span over the number of words (row-length of RA matrix)
        let r_words = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_words.log_2());

        let booleanity_prover_state = BooleanityProverState::new(
            read_addresses,
            EqPolynomial::evals(r_x),
            G,
            &r_words,
            num_words,
        );

        Self {
            prover_state: Some(booleanity_prover_state),
            num_lookups,
            num_words,
            r_x: r_x.to_vec(),
            r_words,
            index: 0,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
    ) -> Self {
        let r_c = sm
            .get_verifier_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(0),
                SumcheckId::PrecompileExecution,
            )
            .0;
        let r_x = r_c.r.split_at(num_lookups.log_2()).0;

        let r_words = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_words.log_2());

        Self {
            prover_state: None,
            num_lookups,
            num_words,
            r_x: r_x.to_vec(),
            r_words,
            index: 0,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2() + self.num_lookups.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.num_words.log_2() {
            // Phase 1: First log(num_words) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(num_lookups) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < self.num_words.log_2() {
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
            if round == self.num_words.log_2() - 1 {
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
            SumcheckId::GatherBooleanity,
        );

        EqPolynomial::mle(
            r,
            &self
                .r_words
                .iter()
                .cloned()
                .rev()
                .chain(self.r_x.iter().cloned().rev())
                .collect::<Vec<F>>(),
        ) * (ra_claim.square() - ra_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
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
            SumcheckId::GatherBooleanity,
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
            SumcheckId::GatherBooleanity,
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

struct RafProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
}

impl<F: JoltField> RafProverState<F> {
    fn initialize(
        // <'a, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
        // TODO(AntoineF4C5): Use prover state to recover values for a/Ra and B
        // state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        F: Vec<F>,
    ) -> Self {
        assert_eq!(F.iter().sum::<F>(), F::one());

        let ra = MultilinearPolynomial::from(F.clone());

        Self { ra }
    }
}

// RafSumcheck proves: a(r_x) = Sum_k( ra(r_x, k) * Id(k) )
// r_x is a challenge vector produced in a sumcheck executed in previous steps
struct RafSumcheck<F: JoltField> {
    prover_state: Option<RafProverState<F>>,
    num_words: usize,
    rv_claim_a: F,
    int: IdentityPolynomial<F>,
    r_x: Vec<F>,
    // Index of the gather instance
    index: usize,
}

impl<F: JoltField> RafSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        F: Vec<F>,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = F.len();

        let (r_x, rv_claim) = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileA(0),
                SumcheckId::PrecompileExecution,
            );
        assert_eq!(r_x.r.len(), num_lookups.log_2());

        let raf_prover_state = RafProverState::initialize(F);
        let int = IdentityPolynomial::new(num_words.log_2());

        Self {
            prover_state: Some(raf_prover_state),
            num_words,
            rv_claim_a: rv_claim,
            int,
            r_x: r_x.r.to_vec(),
            index: 0,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
    ) -> Self {
        let (r_x, rv_claim) = sm
            .get_verifier_accumulator()
            .borrow_mut()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileA(0),
                SumcheckId::PrecompileExecution,
            );
        assert_eq!(r_x.r.len(), num_lookups.log_2());
        let int = IdentityPolynomial::new(num_words.log_2());

        Self {
            prover_state: None,
            num_words,
            rv_claim_a: rv_claim,
            int,
            r_x: r_x.r.to_vec(),
            index: 0,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RafSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
        self.rv_claim_a
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let RafProverState { ra } = self.prover_state.as_ref().unwrap();
        let int = &self.int;

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let int_evals = int.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [ra_evals[0] * int_evals[0], ra_evals[1] * int_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let RafProverState { ra } = self.prover_state.as_mut().unwrap();
        let int = &mut self.int;
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || int.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherRaf,
        );
        let int_claim = self
            .int
            .evaluate(&r.to_vec().into_iter().rev().collect::<Vec<F>>());

        ra_claim * int_claim
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self.prover_state.as_ref().unwrap();

        let ra_claim = prover_state.ra.final_sumcheck_claim();

        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherRaf,
            r_a.into(),
            ra_claim,
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
            SumcheckId::GatherRaf,
            r_a.into(),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::jolt::{pcs::OpeningId, precompiles::gather::test::test_gather_sumcheck};

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero, test_rng};
    use itertools::Itertools;
    use rand_core::RngCore;

    // Creates a random `Gather` instance, and outputs a tuple of (a, B, output)
    fn random_gather(
        // Number of dictionnary entries to recover
        num_lookups: usize,
        // Number of words in the dictionnary to gather from
        num_words: usize,
        // Number of dimensions per word of dictionnary
        word_dim: usize,
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        let mut rng = test_rng();

        let read_addresses: Vec<Fr> = (0..num_lookups)
            .map(|_| Fr::from_u32(rng.next_u32() % num_words as u32))
            .collect();

        let dictionnary: Vec<Fr> = (0..num_words * word_dim)
            .map(|_| Fr::random(&mut rng))
            .collect();

        // Expected output of the gather node: a matrix where each row `i` corresponds to the row at index `a[i]` in B
        let output: Vec<Fr> = read_addresses
            .iter()
            .flat_map(|&index| {
                let index = index.to_u64().unwrap() as usize;
                dictionnary[index * word_dim..(index + 1) * word_dim].to_vec()
            })
            .collect();

        (read_addresses, dictionnary, output)
    }

    struct VirtualOpening {
        poly: VirtualPolynomial,
        sumcheck: SumcheckId,
        point: Vec<Fr>,
        claim: Fr,
    }

    // TODO(AntoineF4C5): Hope is to remove this and have cleaner implementation
    impl VirtualOpening {
        fn insert_in_prover_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
        where
            PT: Transcript,
            CS: CommitmentScheme<Field = Fr>,
        {
            sm.get_prover_accumulator().borrow_mut().append_virtual(
                self.poly,
                self.sumcheck,
                self.point.clone().into(),
                self.claim,
            );
        }
        fn insert_in_verifier_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
        where
            PT: Transcript,
            CS: CommitmentScheme<Field = Fr>,
        {
            sm.get_verifier_accumulator()
                .borrow_mut()
                .openings_mut()
                .insert(
                    OpeningId::Virtual(self.poly, self.sumcheck),
                    (OpeningPoint::default(), self.claim.clone()),
                );

            sm.get_verifier_accumulator().borrow_mut().append_virtual(
                self.poly,
                self.sumcheck,
                self.point.clone().into(),
            );
        }
    }

    fn gather_instances<ProofTranscript, CS>(
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (read_addresses, dictionnary, output): (Vec<Fr>, Vec<Fr>, Vec<Fr>),
    ) -> (
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Prover instance
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Verifier instance
    )
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let num_lookups = read_addresses.len();
        let word_dim = output.len() / num_lookups;
        let _num_words = dictionnary.len() / word_dim;

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduce output matrix to a single evaluation
        let r_c: Vec<Fr> = prover_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());
        assert_eq!(
            r_c,
            verifier_sm
                .get_transcript()
                .borrow_mut()
                .challenge_vector::<Fr>(num_lookups.log_2() + word_dim.log_2())
        );
        let (r_x, _r_y) = r_c.split_at(num_lookups.log_2());

        // Create openings that are inserted in the state manager before creating instances
        let rv_claim_c = MultilinearPolynomial::from(output).evaluate(&r_c);
        let mut virtual_openings = Vec::new();
        virtual_openings.push(VirtualOpening {
            poly: VirtualPolynomial::PrecompileC(0),
            sumcheck: SumcheckId::PrecompileExecution,
            point: r_c.clone(),
            claim: rv_claim_c,
        });

        let rv_claim_a = MultilinearPolynomial::from(read_addresses).evaluate(r_x);
        virtual_openings.push(VirtualOpening {
            poly: VirtualPolynomial::PrecompileA(0),
            sumcheck: SumcheckId::PrecompileExecution,
            point: r_x.to_vec(),
            claim: rv_claim_a,
        });

        virtual_openings.iter().for_each(|opening| {
            opening.insert_in_prover_sm(prover_sm);
            opening.insert_in_verifier_sm(verifier_sm);
        });

        let prover_instances = GatherProof::gather_prover_instances(
            prover_sm,
            read_addresses_usize,
            dictionnary.clone(),
            word_dim,
        );
        let verifier_instances = GatherProof::gather_verifier_instances(
            verifier_sm,
            num_lookups,
            dictionnary.len(),
            word_dim,
        );

        (prover_instances, verifier_instances)
    }

    fn booleanity_instances<ProofTranscript, CS>(
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (read_addresses, dictionnary, output): (Vec<Fr>, Vec<Fr>, Vec<Fr>),
    ) -> (
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Prover instance
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Verifier instance
    )
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let num_lookups = read_addresses.len();
        let word_dim = output.len() / num_lookups;
        let num_words = dictionnary.len() / word_dim;

        let read_addresses: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let r_c: Vec<Fr> = prover_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());

        let _r_c: Vec<Fr> = verifier_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());
        assert_eq!(r_c, _r_c);

        // Create an opening that is inserted in the state manager before
        let rv_claim_c = MultilinearPolynomial::from(output).evaluate(&r_c);
        let virtual_opening = VirtualOpening {
            poly: VirtualPolynomial::PrecompileC(0),
            sumcheck: SumcheckId::PrecompileExecution,
            point: r_c.clone(),
            claim: rv_claim_c,
        };
        virtual_opening.insert_in_prover_sm(prover_sm);
        virtual_opening.insert_in_verifier_sm(verifier_sm);

        let (r_x, _) = r_c.split_at(num_lookups.log_2());

        let E: Vec<Fr> = EqPolynomial::evals(r_x);
        let F: Vec<Fr> = (0..num_words)
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
        assert_eq!(E.len(), num_lookups);
        assert_eq!(F.len(), num_words);

        let prover_booleanity_sumcheck =
            BooleanitySumcheck::new_prover(prover_sm, read_addresses, F);

        let verifier_booleanity_sumcheck =
            BooleanitySumcheck::new_verifier(verifier_sm, num_lookups, num_words);

        (
            vec![Box::new(prover_booleanity_sumcheck)],
            vec![Box::new(verifier_booleanity_sumcheck)],
        )
    }

    // returns instances for the rv and hamming-weight proving sumcheck
    fn rv_hw_instances<ProofTranscript, CS>(
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (read_addresses, dictionnary, output): (Vec<Fr>, Vec<Fr>, Vec<Fr>),
    ) -> (
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Prover instance
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Verifier instance
    )
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        // Number of dictionnary entries to recover
        let num_lookups = read_addresses.len();
        // Number of dimensions per word of dictionnary
        let word_dim = output.len() / num_lookups;
        // Number of words in the dictionnary to gather from
        let num_words = dictionnary.len() / word_dim;

        let (read_addresses, dictionnary, output) = random_gather(num_lookups, num_words, word_dim);

        let read_addresses: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let r_c: Vec<Fr> = prover_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());
        let _r_c: Vec<Fr> = verifier_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());
        assert_eq!(r_c, _r_c);

        let rv_claim = MultilinearPolynomial::from(output).evaluate(&r_c);

        // Create an opening that is inserted in the state manager before
        let virtual_opening = VirtualOpening {
            poly: VirtualPolynomial::PrecompileC(0),
            sumcheck: SumcheckId::PrecompileExecution,
            point: r_c.clone(),
            claim: rv_claim,
        };
        virtual_opening.insert_in_prover_sm(prover_sm);
        virtual_opening.insert_in_verifier_sm(verifier_sm);

        let (r_x, r_y) = r_c.split_at(num_lookups.log_2());

        let E: Vec<Fr> = EqPolynomial::evals(r_x);
        let F: Vec<Fr> = (0..num_words)
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
        assert_eq!(E.len(), num_lookups);
        assert_eq!(F.len(), num_words);

        let eq_r_y = EqPolynomial::evals(r_y);
        let folded_dict: Vec<Fr> = dictionnary
            .chunks(word_dim)
            .map(|word_vector| {
                word_vector
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(d, e)| d * e)
                    .sum()
            })
            .collect();
        assert_eq!(folded_dict.len(), num_words);

        let rv_hw_prover_sumcheck =
            RvHwSumcheck::new_prover(prover_sm, read_addresses, folded_dict, F);

        let rv_hw_verifier_sumcheck =
            RvHwSumcheck::new_verifier(verifier_sm, num_lookups, num_words);

        (
            vec![Box::new(rv_hw_prover_sumcheck)],
            vec![Box::new(rv_hw_verifier_sumcheck)],
        )
    }

    // returns instances for the raf-evaluation proving sumcheck
    fn raf_instances<ProofTranscript, CS>(
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (read_addresses, dictionnary, output): (Vec<Fr>, Vec<Fr>, Vec<Fr>),
    ) -> (
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Prover instance
        Vec<Box<dyn SumcheckInstance<Fr>>>, // Verifier instance
    )
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        // Number of dictionnary entries to recover
        let num_lookups = read_addresses.len();
        // Number of dimensions per word of dictionnary
        let word_dim = output.len() / num_lookups;
        // Number of words in the dictionnary to gather from
        let num_words = dictionnary.len() / word_dim;

        let (read_addresses, _dictionnary, _output) =
            random_gather(num_lookups, num_words, word_dim);

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let r_x: Vec<Fr> = prover_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2());
        let _r_x: Vec<Fr> = verifier_sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2());
        assert_eq!(r_x, _r_x);

        let rv_claim_a = MultilinearPolynomial::from(read_addresses).evaluate(&r_x);

        // Create an opening that is inserted in the state manager before
        let virtual_opening = VirtualOpening {
            poly: VirtualPolynomial::PrecompileA(0),
            sumcheck: SumcheckId::PrecompileExecution,
            point: r_x.clone(),
            claim: rv_claim_a,
        };
        virtual_opening.insert_in_prover_sm(prover_sm);
        virtual_opening.insert_in_verifier_sm(verifier_sm);

        let E: Vec<Fr> = EqPolynomial::evals(&r_x);
        let F: Vec<Fr> = (0..num_words)
            .into_par_iter()
            .map(|k| {
                read_addresses_usize
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum()
            })
            .collect();
        assert_eq!(E.len(), num_lookups);
        assert_eq!(F.len(), num_words);

        let raf_prover_sumcheck = RafSumcheck::new_prover(prover_sm, read_addresses_usize, F);

        let raf_verifier_sumcheck = RafSumcheck::new_verifier(verifier_sm, num_lookups, num_words);

        (
            vec![Box::new(raf_prover_sumcheck)],
            vec![Box::new(raf_verifier_sumcheck)],
        )
    }

    #[test]
    fn test_gather_proof() {
        // Number of words to recover from the dictionnary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionnary
        const WORD_DIM: usize = 1 << 3;

        let tensors = random_gather(NUM_LOOKUPS, NUM_WORDS, WORD_DIM);

        let _ = test_gather_sumcheck(gather_instances, tensors.clone());
    }

    #[test]
    fn test_booleanity() {
        // Number of words to recover from the dictionnary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionnary
        const WORD_DIM: usize = 1 << 3;

        let tensors = random_gather(NUM_LOOKUPS, NUM_WORDS, WORD_DIM);

        let openings = test_gather_sumcheck(booleanity_instances, tensors.clone());
        let (r, claim) = openings
            .get(&OpeningId::Virtual(
                VirtualPolynomial::GatherRa(0),
                SumcheckId::GatherBooleanity,
            ))
            .expect("GatherRa(0) should be set");

        let (read_addresses, _dictionnary, _output) = tensors;

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        let ra: Vec<Fr> = read_addresses_usize
            .iter()
            .flat_map(|&address| {
                let mut one_hot = vec![Fr::zero(); NUM_WORDS];
                one_hot[address] = Fr::one();
                one_hot
            })
            .collect();

        let expected_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
        assert_eq!(*claim, expected_claim);
    }

    #[test]
    fn test_rv_hw() {
        // Number of words to recover from the dictionnary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionnary
        const WORD_DIM: usize = 1 << 3;

        let tensors = random_gather(NUM_LOOKUPS, NUM_WORDS, WORD_DIM);

        let openings = test_gather_sumcheck(rv_hw_instances, tensors.clone());

        let (read_addresses, dictionnary, _output) = tensors;

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        let ra: Vec<Fr> = read_addresses_usize
            .iter()
            .flat_map(|&address| {
                let mut one_hot = vec![Fr::zero(); NUM_WORDS];
                one_hot[address] = Fr::one();
                one_hot
            })
            .collect();

        let (r, claim) = openings
            .get(&OpeningId::Virtual(
                VirtualPolynomial::GatherRa(0),
                SumcheckId::GatherRvHw,
            ))
            .expect("GatherRa(0) should be set");

        let exp_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
        assert_eq!(*claim, exp_claim);

        let (r, claim) = openings
            .get(&OpeningId::Virtual(
                VirtualPolynomial::PrecompileB(0),
                SumcheckId::GatherRvHw,
            ))
            .expect("PrecompileB(0) should be set");

        let exp_claim = MultilinearPolynomial::from(dictionnary).evaluate(&r.r);
        assert_eq!(*claim, exp_claim);
    }

    #[test]
    fn test_raf_eval() {
        // Number of words to recover from the dictionnary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionnary
        const WORD_DIM: usize = 1 << 3;

        let tensors = random_gather(NUM_LOOKUPS, NUM_WORDS, WORD_DIM);

        let openings = test_gather_sumcheck(raf_instances, tensors.clone());

        let (read_addresses, _dictionnary, _output) = tensors;

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .collect();

        let ra: Vec<Fr> = read_addresses_usize
            .iter()
            .flat_map(|&address| {
                let mut one_hot = vec![Fr::zero(); NUM_WORDS];
                one_hot[address] = Fr::one();
                one_hot
            })
            .collect();

        let (r, claim) = openings
            .get(&OpeningId::Virtual(
                VirtualPolynomial::GatherRa(0),
                SumcheckId::GatherRaf,
            ))
            .expect("GatherRa(0) should be set");

        let exp_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
        assert_eq!(*claim, exp_claim);
    }
}
