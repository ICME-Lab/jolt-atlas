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
// ReadValueSumcheck proves: C(r_x, r_y) = Sum_k( ra(r_x, k) * B(k, r_y) )
//  - Read Value ensures that the output of this instruction is indeed equal to the matrix-multiplication of ra and b,
//      where ra is claimed to be the one-hot encoding matrix of input vector A. (row `i` of ra is one-hot encoding of `A[i]`)
//
// BooleanitySumcheck proves:  0 = Sum_k,t( eq(r_x, t) * eq(r_y, k) * (ra(t, k)² - ra(t, k)) )
//  - Booleanity ensures that each value in ra is in the set {0, 1}.
//
// RafSumcheck proves: a(r_x) = Sum_k( ra(r_x, k) * Id(k) )
//  - Raf Evaluation computes dot product of each row of ra with the identity polynomial.
//  This and Hamming Weight, Booleanity ensures that ra's row `i` is the one-hot encoding of `A[i]`
//
// HammingBooleanitySumcheck proves: 0 = Sum_k( eq(r, k) * (hw(k)² - hw(k)) )
// - Hamming Booleanity ensures that the hamming weight of ra is comprise in {0, 1} for each row (i.e. that each row either encodes a lookup, or no lookup (padding))
//
// HwSumcheck proves: hw(r_h) = Sum_k( ra(r_h, k) )
//  - Hamming Weight verifies the evaluation on hw claimed in HammingBooleanity sumcheck
//  - Since HwSumcheck's input claim is the evaluation of hw at r_h, computed in HammingBooleanitySumcheck, this sumcheck cannot be batched with the others
//
//
// Putting it in the context of the Gather instructions:
// id   | dims                      | description                                                                                   | name in sumchecks
// A    | [num_lookups]             | first input vector, where `A[i]` is the index of the values to retrieve from the second input | read_addresses
// B    | [num_words * word_dim]    | second input, a matrix where each row holds the values to be retrieved                        | dictionary
// C    | [num_lookups * word_dim]  | output, a matrix where each row holds the retrieved values from B at index given in A         | output
// ra   | [num_lookups * num_words] | used by sumchecks, each row holds the one-hot encoding of value held in A                     | ra

struct ReadValueProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
    dict_folded: MultilinearPolynomial<F>,
}

impl<F: JoltField> ReadValueProverState<F> {
    fn new(dict_folded: Vec<F>, ra_folded: Vec<F>) -> Self {
        let ra = MultilinearPolynomial::from(ra_folded);
        let dict_folded = MultilinearPolynomial::from(dict_folded);

        Self { ra, dict_folded }
    }
}

// ReadValueSumcheck proves: C(r_x, r_y) = Sum_k( ra(r_x, k) * B(k, r_y) )
//  - Read Value ensures that the output of this instruction is indeed equal to the matrix-multiplication of ra and b,
//      where ra is claimed to be the one-hot encoding matrix of input vector A. (row `i` of ra is one-hot encoding of `A[i]`)
pub struct ReadValueSumcheck<F: JoltField> {
    prover_state: Option<ReadValueProverState<F>>,
    // Dimension over which sumcheck is ran
    num_words: usize,
    rv_claim: F,
    r_x: Vec<F>,
    r_y: Vec<F>,
    // Index of the gather instance
    index: usize,
}

impl<F: JoltField> ReadValueSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        dictionary_folded: Vec<F>,
        ra_folded: Vec<F>,
        index: usize,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = dictionary_folded.len();

        let (r_c, rv_claim_c) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
        );
        let (r_x, r_y) = r_c.r.split_at(num_lookups.log_2());

        let rv_prover_state = ReadValueProverState::new(dictionary_folded, ra_folded);

        Self {
            prover_state: Some(rv_prover_state),
            rv_claim: rv_claim_c,
            num_words,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
            index,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
        index: usize,
    ) -> Self {
        let (r_c, rv_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
        );
        let (r_x, r_y) = r_c.r.split_at(num_lookups.log_2());

        Self {
            prover_state: None,
            num_words,
            rv_claim,
            r_x: r_x.to_vec(),
            r_y: r_y.to_vec(),
            index,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ReadValueSumcheck<F> {
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ReadValueProverState { ra, dict_folded } = self.prover_state.as_ref().unwrap();

        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let dict_evals = dict_folded.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [ra_evals[0] * dict_evals[0], ra_evals[1] * dict_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let ReadValueProverState { ra, dict_folded } = self.prover_state.as_mut().unwrap();
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
            SumcheckId::GatherRv,
        );
        let (_, dict_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
        );

        ra_claim * dict_claim
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
            SumcheckId::GatherRv,
            r_a.into(),
            ra_claim,
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
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
            SumcheckId::GatherRv,
            r_a.into(),
        );

        let r_b = [opening_point.r, self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
        );
    }
}

struct HammingBooleanityProverState<F: JoltField> {
    hw: MultilinearPolynomial<F>,
    eq_r_x: MultilinearPolynomial<F>,
}

impl<F: JoltField> HammingBooleanityProverState<F> {
    fn new(hw: Vec<F>, eq_r_x: Vec<F>) -> Self {
        let hw = MultilinearPolynomial::from(hw);
        let eq_r_x = MultilinearPolynomial::from(eq_r_x);

        Self { hw, eq_r_x }
    }
}

// HammingBooleanitySumcheck proves: 0 = Sum_k( eq(r, k) * (hw(k)² - hw(k)) )
// - Hamming Booleanity ensures that the hamming weight of ra is comprise in {0, 1} for each row (i.e. that each row either encodes a lookup, or no lookup (padding))
pub struct HammingBooleanitySumcheck<F: JoltField> {
    prover_state: Option<HammingBooleanityProverState<F>>,
    // Dimension over which sumcheck is ran
    num_lookups: usize,
    r_x: Vec<F>,
    // Index of the gather instance
    index: usize,
}

impl<F: JoltField> HammingBooleanitySumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        index: usize,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let mut hw = vec![F::one(); num_lookups];
        hw.resize(num_lookups.next_power_of_two(), F::zero());

        let (r_x, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );

        let eq_r_x = EqPolynomial::evals(&r_x.r);
        let hw_boolean_prover_state = HammingBooleanityProverState::new(hw, eq_r_x);

        Self {
            prover_state: Some(hw_boolean_prover_state),
            num_lookups,
            r_x: r_x.r,
            index,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        index: usize,
    ) -> Self {
        let (r_x, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );

        Self {
            prover_state: None,
            num_lookups,
            r_x: r_x.r,
            index,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HammingBooleanitySumcheck<F> {
    #[inline(always)]
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_lookups.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let HammingBooleanityProverState { hw, eq_r_x } = self.prover_state.as_ref().unwrap();

        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 3] = (0..hw.len() / 2)
            .into_par_iter()
            .map(|i| {
                let hw_evals = hw.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let eq_r_x = eq_r_x.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    eq_r_x[0] * (hw_evals[0].square() - hw_evals[0]),
                    eq_r_x[1] * (hw_evals[1].square() - hw_evals[1]),
                    eq_r_x[2] * (hw_evals[2].square() - hw_evals[2]),
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

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let HammingBooleanityProverState { hw, eq_r_x } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || hw.bind_parallel(r_j, BindingOrder::LowToHigh),
            || eq_r_x.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let (_, hw_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherHammingWeight(self.index),
            SumcheckId::GatherHB,
        );

        let eq = EqPolynomial::mle(&self.r_x.iter().cloned().rev().collect::<Vec<F>>(), r);

        eq * (hw_claim.square() - hw_claim)
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

        let hw_claim = prover_state.hw.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherHammingWeight(self.index),
            SumcheckId::GatherHB,
            opening_point,
            hw_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherHammingWeight(self.index),
            SumcheckId::GatherHB,
            opening_point,
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

// BooleanitySumcheck proves:  0 = Sum_k,t( eq(r_x, t) * eq(r_y, k) * (ra(t, k)² - ra(t, k)) )
//  - Booleanity ensures that each value in ra is in the set {0, 1}.
pub struct BooleanitySumcheck<F: JoltField> {
    prover_state: Option<BooleanityProverState<F>>,
    num_lookups: usize,
    num_words: usize,
    r_x: Vec<F>,
    r_words: Vec<F>,
    index: usize,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        G: Vec<F>,
        index: usize,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = G.len();

        let (r_x, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );

        // Generate a random challenge to complete with r_x for RA booleanity sumcheck:
        // r_x spans over the input length (column-length of RA matrix),
        // this random challenge will span over the number of words (row-length of RA matrix)
        let r_words = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_words.log_2());

        let booleanity_prover_state = BooleanityProverState::new(
            read_addresses,
            EqPolynomial::evals(&r_x.r),
            G,
            &r_words,
            num_words,
        );

        Self {
            prover_state: Some(booleanity_prover_state),
            num_lookups,
            num_words,
            r_x: r_x.r.to_vec(),
            r_words,
            index,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
        index: usize,
    ) -> Self {
        let (r_x, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );

        let r_words = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_words.log_2());

        Self {
            prover_state: None,
            num_lookups,
            num_words,
            r_x: r_x.r.to_vec(),
            r_words,
            index,
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

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.num_words.log_2() {
            // Phase 1: First log(num_words) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(num_lookups) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all)]
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
                    let mut coeffs: Vec<F> = std::mem::take(&mut read_addresses)
                        .into_par_iter()
                        .map(|j| f_ref[j])
                        .collect();
                    coeffs.resize(coeffs.len().next_power_of_two(), F::zero());
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
    fn new(ra_folded: Vec<F>) -> Self {
        let ra = MultilinearPolynomial::from(ra_folded);

        Self { ra }
    }
}

// RafSumcheck proves: a(r_x) = Sum_k( ra(r_x, k) * Id(k) )
//  - Raf Evaluation computes dot product of each row of ra with the identity polynomial.
//  This and Hamming Weight, Booleanity ensures that ra's row `i` is the one-hot encoding of `A[i]`
pub struct RafSumcheck<F: JoltField> {
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
        ra_folded: Vec<F>,
        index: usize,
    ) -> Self {
        let num_lookups = read_addresses.len();
        let num_words = ra_folded.len();

        let (r_x, rv_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );
        assert_eq!(r_x.r.len(), num_lookups.log_2());

        let raf_prover_state = RafProverState::new(ra_folded);
        let int = IdentityPolynomial::new(num_words.log_2());

        Self {
            prover_state: Some(raf_prover_state),
            num_words,
            rv_claim_a: rv_claim,
            int,
            r_x: r_x.r.to_vec(),
            index,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
        index: usize,
    ) -> Self {
        let (r_x, rv_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
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
            index,
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
            .evaluate(&r.iter().copied().rev().collect::<Vec<F>>());

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

        let r_a = [self.r_x.clone(), opening_point.r].concat();
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
        let r_a = [self.r_x.clone(), opening_point.r].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherRaf,
            r_a.into(),
        );
    }
}

struct HwProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
}

impl<F: JoltField> HwProverState<F> {
    fn new(ra_folded: Vec<F>) -> Self {
        let ra = MultilinearPolynomial::from(ra_folded);

        Self { ra }
    }
}

// HwSumcheck proves: HW(r_x) = Sum_k( ra(r_x, k) )
// r_x is a challenge produced in a sumcheck executed in previous steps
pub struct HwSumcheck<F: JoltField> {
    prover_state: Option<HwProverState<F>>,
    // Dimension over which sumcheck is ran
    num_words: usize,
    hw_claim: F,
    r_x: Vec<F>,
    // Index of the gather instance
    index: usize,
}

impl<F: JoltField> HwSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_words: usize,
        ra_folded: Vec<F>,
        index: usize,
    ) -> Self {
        let (r_x, hw_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::GatherHammingWeight(index),
            SumcheckId::GatherHB,
        );

        let rv_hw_prover_state = HwProverState::new(ra_folded);

        Self {
            prover_state: Some(rv_hw_prover_state),
            hw_claim,
            num_words,
            r_x: r_x.r.to_vec(),
            index,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_words: usize,
        index: usize,
    ) -> Self {
        let (r_x, hw_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::GatherHammingWeight(index),
            SumcheckId::GatherHB,
        );

        Self {
            prover_state: None,
            num_words,
            hw_claim,
            r_x: r_x.r.to_vec(),
            index,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HwSumcheck<F> {
    #[inline(always)]
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }

    fn input_claim(&self) -> F {
        self.hw_claim
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let HwProverState { ra } = self.prover_state.as_ref().unwrap();

        let degree = <Self as SumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 1] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [ra_evals[0]]
            })
            .reduce(|| [F::zero(); 1], |running, new| [running[0] + new[0]]);
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let HwProverState { ra } = self.prover_state.as_mut().unwrap();
        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherHW,
        );

        ra_claim
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

        let r_a = [self.r_x.clone(), opening_point.r].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherHW,
            r_a.into(),
            ra_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_a = [self.r_x.clone(), opening_point.r].concat();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::GatherRa(self.index),
            SumcheckId::GatherHW,
            r_a.into(),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::jolt::{
        pcs::OpeningId,
        precompiles::gather::{
            compute_ra_evals,
            test::{TestInstance, VirtualOpening, random_gather, test_gather_sumcheck},
        },
    };

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use itertools::Itertools;
    use rand::rngs::StdRng;

    fn booleanity_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, num_words, _) = test_io.dims();
        let (read_addresses, _, _) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        let virtual_opening = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(read_addresses.clone()),
            None,
        );

        let ra_folded = compute_ra_evals(
            &virtual_opening.r,
            &read_addresses_usize,
            num_words.next_power_of_two(),
        );

        let prover_booleanity_sumcheck =
            BooleanitySumcheck::new_prover(prover_sm, read_addresses_usize, ra_folded, index);

        let verifier_booleanity_sumcheck = BooleanitySumcheck::new_verifier(
            verifier_sm,
            num_lookups.next_power_of_two(),
            num_words.next_power_of_two(),
            index,
        );

        TestInstance::new(
            Box::new(prover_booleanity_sumcheck),
            Box::new(verifier_booleanity_sumcheck),
            vec![virtual_opening],
            test_io,
        )
    }

    // returns instances for the rv and hamming-weight proving sumcheck
    fn rv_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, num_words, word_dim) = test_io.dims();
        let (read_addresses, dictionary, output) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let virtual_opening = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(output.clone()),
            None,
        );

        let (r_x, r_y) = virtual_opening.r.split_at(num_lookups.log_2());

        let ra_folded = compute_ra_evals(r_x, &read_addresses_usize, num_words.next_power_of_two());

        let eq_r_y = EqPolynomial::evals(r_y);
        let folded_dict: Vec<Fr> = dictionary
            .chunks(word_dim.next_power_of_two())
            .map(|word_vector| {
                word_vector
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(d, e)| d * e)
                    .sum()
            })
            .collect();
        assert_eq!(folded_dict.len(), num_words.next_power_of_two());

        let rv_prover_sumcheck = ReadValueSumcheck::new_prover(
            prover_sm,
            read_addresses_usize,
            folded_dict,
            ra_folded,
            index,
        );

        let rv_verifier_sumcheck = ReadValueSumcheck::new_verifier(
            verifier_sm,
            num_lookups.next_power_of_two(),
            num_words.next_power_of_two(),
            index,
        );

        TestInstance::new(
            Box::new(rv_prover_sumcheck),
            Box::new(rv_verifier_sumcheck),
            vec![virtual_opening],
            test_io,
        )
    }

    // returns instances for the hamming booleanity
    fn hb_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, _, _) = test_io.dims();
        let (read_addresses, _, _) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let virtual_opening = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(read_addresses.clone()),
            None,
        );

        let hb_prover_sumcheck =
            HammingBooleanitySumcheck::new_prover(prover_sm, read_addresses_usize, index);

        let hb_verifier_sumcheck = HammingBooleanitySumcheck::new_verifier(
            verifier_sm,
            num_lookups.next_power_of_two(),
            index,
        );

        TestInstance::new(
            Box::new(hb_prover_sumcheck),
            Box::new(hb_verifier_sumcheck),
            vec![virtual_opening],
            test_io,
        )
    }

    // returns instances for the raf-evaluation proving sumcheck
    fn raf_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, num_words, _) = test_io.dims();
        let (read_addresses, _, _) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let virtual_opening = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(read_addresses.clone()),
            None,
        );

        let ra_folded = compute_ra_evals(
            &virtual_opening.r,
            &read_addresses_usize,
            num_words.next_power_of_two(),
        );

        let raf_prover_sumcheck =
            RafSumcheck::new_prover(prover_sm, read_addresses_usize, ra_folded, index);

        let raf_verifier_sumcheck = RafSumcheck::new_verifier(
            verifier_sm,
            num_lookups.next_power_of_two(),
            num_words.next_power_of_two(),
            index,
        );

        TestInstance::new(
            Box::new(raf_prover_sumcheck),
            Box::new(raf_verifier_sumcheck),
            vec![virtual_opening],
            test_io,
        )
    }

    // returns instances for the hamming-weight proving sumcheck
    fn hw_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, num_words, _) = test_io.dims();
        let (read_addresses, _, _) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        let mut hw = vec![Fr::one(); num_lookups];
        hw.resize(num_lookups.next_power_of_two(), Fr::zero());

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduces to a single evaluation over output dimensions
        let virtual_opening = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::GatherHammingWeight(index),
            SumcheckId::GatherHB,
            MultilinearPolynomial::from(hw),
            None,
        );

        let ra_folded = compute_ra_evals(
            &virtual_opening.r,
            &read_addresses_usize,
            num_words.next_power_of_two(),
        );

        let hw_prover_sumcheck =
            HwSumcheck::new_prover(prover_sm, num_words.next_power_of_two(), ra_folded, index);

        let hw_verifier_sumcheck =
            HwSumcheck::new_verifier(verifier_sm, num_words.next_power_of_two(), index);

        TestInstance::new(
            Box::new(hw_prover_sumcheck),
            Box::new(hw_verifier_sumcheck),
            vec![virtual_opening],
            test_io,
        )
    }

    #[test]
    fn test_booleanity() {
        // Number of words to recover from the dictionary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) = test_gather_sumcheck(
            booleanity_instances,
            (NUM_LOOKUPS, NUM_WORDS, WORD_DIM),
            123456,
            10,
        );

        // Verify openings
        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, _, _) = instance.values();
            let (num_lookups, num_words, _) = instance.dims();

            let read_addresses_usize: Vec<usize> = read_addresses
                .iter()
                .map(|e| e.to_u64().unwrap() as usize)
                .take(num_lookups)
                .collect();

            let mut ra: Vec<Fr> = read_addresses_usize
                .iter()
                .flat_map(|&address| {
                    let mut one_hot = vec![Fr::zero(); num_words.next_power_of_two()];
                    one_hot[address] = Fr::one();
                    one_hot
                })
                .collect();
            ra.resize(ra.len().next_power_of_two(), Fr::zero());

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherBooleanity,
                ))
                .expect("GatherRa(index) should be set");

            let expected_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
            assert_eq!(*claim, expected_claim);
        }
    }

    #[test]
    fn test_rv() {
        // Number of words to recover from the dictionary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) =
            test_gather_sumcheck(rv_instances, (NUM_LOOKUPS, NUM_WORDS, WORD_DIM), 123456, 10);

        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, dictionary, _) = instance.values();
            let (num_lookups, num_words, _) = instance.dims();

            let read_addresses_usize: Vec<usize> = read_addresses
                .iter()
                .map(|e| e.to_u64().unwrap() as usize)
                .take(num_lookups)
                .collect();

            let mut ra: Vec<Fr> = read_addresses_usize // get to pow2 size
                .iter()
                .flat_map(|&address| {
                    let mut one_hot = vec![Fr::zero(); num_words.next_power_of_two()];
                    one_hot[address] = Fr::one();
                    one_hot
                })
                .collect();
            ra.resize(ra.len().next_power_of_two(), Fr::zero());

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRv,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::PrecompileB(i),
                    SumcheckId::PrecompileExecution,
                ))
                .expect("PrecompileB(index) should be set");

            let exp_claim = MultilinearPolynomial::from(dictionary.clone()).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }

    #[test]
    fn test_hb() {
        // Number of words to recover from the dictionary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) =
            test_gather_sumcheck(hb_instances, (NUM_LOOKUPS, NUM_WORDS, WORD_DIM), 123456, 10);

        for (i, instance) in instances.into_iter().enumerate() {
            let (num_lookups, _, _) = instance.dims();

            let mut hw: Vec<Fr> = vec![Fr::one(); num_lookups];
            hw.resize(num_lookups.next_power_of_two(), Fr::zero());

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherHammingWeight(i),
                    SumcheckId::GatherHB,
                ))
                .expect("GatherHammingBooleanity(index) should be set");

            let exp_claim = MultilinearPolynomial::from(hw).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }

    #[test]
    fn test_raf_eval() {
        // Number of words to recover from the dictionary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) = test_gather_sumcheck(
            raf_instances,
            (NUM_LOOKUPS, NUM_WORDS, WORD_DIM),
            123456,
            10,
        );

        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, _, _) = instance.values();
            let (num_lookups, num_words, _) = instance.dims();

            let read_addresses_usize: Vec<usize> = read_addresses
                .iter()
                .map(|e| e.to_u64().unwrap() as usize)
                .take(num_lookups)
                .collect();

            let mut ra: Vec<Fr> = read_addresses_usize
                .iter()
                .flat_map(|&address| {
                    let mut one_hot = vec![Fr::zero(); num_words.next_power_of_two()];
                    one_hot[address] = Fr::one();
                    one_hot
                })
                .collect();
            ra.resize(ra.len().next_power_of_two(), Fr::zero());

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRaf,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }

    #[test]
    fn test_hw() {
        // Number of words to recover from the dictionary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) =
            test_gather_sumcheck(hw_instances, (NUM_LOOKUPS, NUM_WORDS, WORD_DIM), 123456, 10);

        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, _, _) = instance.values();
            let (num_lookups, num_words, _) = instance.dims();
            let read_addresses_usize: Vec<usize> = read_addresses
                .iter()
                .map(|e| e.to_u64().unwrap() as usize)
                .take(num_lookups)
                .collect();

            let mut ra: Vec<Fr> = read_addresses_usize
                .iter()
                .flat_map(|&address| {
                    let mut one_hot = vec![Fr::zero(); num_words.next_power_of_two()];
                    one_hot[address] = Fr::one();
                    one_hot
                })
                .collect();
            ra.resize(ra.len().next_power_of_two(), Fr::zero());

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherHW,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = MultilinearPolynomial::from(ra).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }
}
