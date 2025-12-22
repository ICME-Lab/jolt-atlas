use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
        unipoly::UniPoly,
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};
use std::{cell::RefCell, rc::Rc};

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    precompiles::gather::shout::{BooleanitySumcheck, RafSumcheck, RvHwSumcheck},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

pub mod axis_0;
pub mod shout;
#[cfg(test)]
pub mod test;

// Batching the different gather execution sumchecks. (Booleanity, Raf-Eval, ReadValue and HammingWeight)
// I chosed not to use existing BatchedSumcheck::prove workflow, as it would still require me to send 3 sumcheck instances to the precompiledag.
// I prefered batching all instances to one, rather than keeping 3 instances and just batching the proof.
// This allows to keep the number of precompile execution instances equal to the number of precompile instructions in trace.
// (Otherwise we would have to set, in precompiles params, num_instances = num_precompiles + 2 * num_gathers) which I wanted to avoid.
//
// I could have also created a large single Sumcheck instance where I concatenate code for executing all three required sumchecks.
// However this would have removed the simplicity of having 3 differents sumchecks which we can individually test.
// For example, `jolt::executor::ReadRafSumcheck` is a case of batching several sumchecks to one, making it quite hard to test
//
// NOTE: For now this looks messy, but I am pretty sure if we iterate on this we could actually create a new way to batch sumchecks,
// while keeping simple individual sumchecks to audit/test
pub struct ExecutionProof {}

impl ExecutionProof {
    #[tracing::instrument(skip_all, name = "ShoutProof::prove")]
    pub fn gather_prover_instance<
        'a,
        F: JoltField,
        PT: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'a, F, PT, PCS>,
        read_addresses: Vec<usize>,
        dictionnary: Vec<F>,
        word_dim: usize,
        index: usize,
    ) -> Box<dyn SumcheckInstance<F>> {
        let num_lookups = read_addresses.len();
        let num_words = dictionnary.len() / word_dim; // dictionnary is a num_words * word_dim matrix

        // Recover challenge built during previous sumcheck
        let (r_c, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
        );

        let (r_x, r_y) = r_c.split_at(num_lookups.log_2());
        assert_eq!(r_y.len(), word_dim.log_2());

        let F = compute_ra_evals(&r_x.r, &read_addresses, num_words);

        let eq_r_y = EqPolynomial::evals(&r_y.r);
        // dictionnary, evaluated at column r_y
        let dictionnary_folded: Vec<F> = dictionnary
            .chunks(word_dim)
            .map(|B_chunk| {
                B_chunk
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(&b, &e)| b * e)
                    .sum()
            })
            .collect();
        assert_eq!(dictionnary_folded.len(), num_words);

        let execution_instance =
            ExecutionSumcheck::init_prover(sm, read_addresses, dictionnary_folded, F, index);

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Box::new(execution_instance)
    }

    pub fn gather_verifier_instance<
        'a,
        F: JoltField,
        PT: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'a, F, PT, PCS>,
        num_lookups: usize,
        dictionnary_length: usize,
        word_dim: usize,
        index: usize,
    ) -> Box<dyn SumcheckInstance<F>> {
        let num_words = dictionnary_length / word_dim;

        let execution_instance =
            ExecutionSumcheck::init_verifier(sm, num_lookups, num_words, index);

        Box::new(execution_instance)
    }
}

pub struct ExecutionSumcheck<F: JoltField> {
    booleanity: BooleanitySumcheck<F>,
    raf: RafSumcheck<F>,
    rv_hw: RvHwSumcheck<F>,
    // folding challenge for different sumchecks
    claims: (F, F, F),
    unipolys: (UniPoly<F>, UniPoly<F>, UniPoly<F>),
    gamma: F,
    gamma_sqr: F,
    // Number of variables for resp, lookups and words dimensions of sumcheck
    lookups_vars: usize,
    words_vars: usize,
}

impl<F: JoltField> ExecutionSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let final_memory_state = sm.get_val_final();
        let (pp, ..) = sm.get_prover_data();
        let pp = &pp.shared.precompiles.instances[index];

        let num_lookups = pp.a_dims[0];
        let num_words = pp.b_dims[0];
        let word_dim = pp.b_dims[1];

        let read_addresses = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let read_addresses_usize: Vec<usize> = read_addresses.iter().map(|&x| x as usize).collect();
        let dictionnary = pp.extract_rv(final_memory_state, |m| &m.b_addr);
        let output = pp.extract_rv(final_memory_state, |m| &m.c_addr);

        let r_c: Vec<F> = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());

        let (r_x, r_y) = r_c.split_at(num_lookups.log_2());

        // Create openings that are inserted in the state manager before creating instances
        let rv_claim_c = MultilinearPolynomial::from(output).evaluate(&r_c);
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.clone().into(),
            rv_claim_c,
        );

        let rv_claim_a = MultilinearPolynomial::from(read_addresses).evaluate(r_x);
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            r_x.to_vec().into(),
            rv_claim_a,
        );
        assert_eq!(r_y.len(), word_dim.log_2());

        let F = compute_ra_evals(r_x, &read_addresses_usize, num_words);

        let eq_r_y = EqPolynomial::evals(r_y);
        // dictionnary, evaluated at column r_y
        let dictionnary_folded: Vec<F> = dictionnary
            .chunks(word_dim)
            .map(|B_chunk| {
                B_chunk
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(&b, &e)| F::from_i64(b) * e)
                    .sum()
            })
            .collect();
        assert_eq!(dictionnary_folded.len(), num_words);

        Self::init_prover(sm, read_addresses_usize, dictionnary_folded, F, index)
    }

    pub fn init_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        dictionnary_folded: Vec<F>,
        F: Vec<F>,
        index: usize,
    ) -> Self {
        let lookups_vars = read_addresses.len().log_2();
        let words_vars = dictionnary_folded.len().log_2();

        let booleanity =
            BooleanitySumcheck::new_prover(sm, read_addresses.clone(), F.clone(), index);
        let raf = RafSumcheck::new_prover(sm, read_addresses.clone(), F.clone(), index);
        let rv_hw = RvHwSumcheck::new_prover(sm, read_addresses, dictionnary_folded, F, index);
        let claims = (
            booleanity.input_claim(),
            raf.input_claim(),
            rv_hw.input_claim(),
        );

        let gamma = sm.get_transcript().borrow_mut().challenge_scalar();
        let gamma_sqr = gamma * gamma;

        Self {
            booleanity,
            raf,
            rv_hw,
            claims,
            unipolys: (UniPoly::zero(), UniPoly::zero(), UniPoly::zero()),
            gamma,
            gamma_sqr,
            lookups_vars,
            words_vars,
        }
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let (pp, ..) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];

        let num_lookups = pp.a_dims[0];
        let num_words = pp.b_dims[0];
        let word_dim = pp.b_dims[1];

        let r_c: Vec<F> = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups.log_2() + word_dim.log_2());

        let (r_x, _r_y) = r_c.split_at(num_lookups.log_2());

        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.clone().into(),
        );

        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            r_x.to_vec().into(),
        );

        Self::init_verifier(sm, num_lookups, num_words, index)
    }

    pub fn init_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
        index: usize,
    ) -> Self {
        let booleanity = BooleanitySumcheck::new_verifier(sm, num_lookups, num_words, index);
        let raf = RafSumcheck::new_verifier(sm, num_lookups, num_words, index);
        let rv_hw = RvHwSumcheck::new_verifier(sm, num_lookups, num_words, index);

        let claims = (
            booleanity.input_claim(),
            raf.input_claim(),
            rv_hw.input_claim(),
        );

        let gamma = sm.get_transcript().borrow_mut().challenge_scalar();
        let gamma_sqr = gamma * gamma;

        Self {
            booleanity,
            raf,
            rv_hw,
            claims,
            unipolys: (UniPoly::zero(), UniPoly::zero(), UniPoly::zero()),
            gamma,
            gamma_sqr,
            lookups_vars: num_lookups.log_2(),
            words_vars: num_words.log_2(),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.lookups_vars + self.words_vars
    }

    fn input_claim(&self) -> F {
        self.booleanity.input_claim()
            + F::from_u32(self.lookups_vars.pow2() as u32)
                * (self.gamma * self.raf.input_claim() + self.gamma_sqr * self.rv_hw.input_claim())
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.lookups_vars {
            // Not yet to raf and rv_hw's variables, only binding booleanity sumcheck
            let mut prover_message = self
                .booleanity
                .compute_prover_message(round, _previous_claim);

            // create and store Univariate Polynomial of current round
            let mut a_evals = prover_message.clone();
            a_evals.insert(1, self.claims.0);

            // raf and rv_hw are not yet getting bound, for now univariate polynomials are constant
            let scaling_pow = self.lookups_vars - round - 1;
            let b_const = self.raf.input_claim().mul_pow_2(scaling_pow);
            let c_const = self.rv_hw.input_claim().mul_pow_2(scaling_pow);

            // Store booleanity sumcheck poly to later compute claim
            let a = UniPoly::from_evals(&a_evals);
            self.unipolys.0 = a;

            // raf and rv_hw polys are constant, hence we just add the gamma-weighted constants to the booleanity evaluations
            prover_message.iter_mut().for_each(|eval| {
                *eval += b_const * self.gamma + c_const * self.gamma_sqr;
            });
            prover_message
        } else {
            let boolean = self
                .booleanity
                .compute_prover_message(round, _previous_claim);
            let mut raf = self.raf.compute_prover_message(round, _previous_claim);
            let mut rv_hw = self.rv_hw.compute_prover_message(round, _previous_claim);
            let mut a = boolean.clone();
            let mut b = raf.clone();
            let mut c = rv_hw.clone();

            // insert eval[1]
            a.insert(1, self.claims.0 - a[0]);
            b.insert(1, self.claims.1 - b[0]);
            c.insert(1, self.claims.2 - c[0]);

            self.unipolys = (
                UniPoly::from_evals(&a),
                UniPoly::from_evals(&b),
                UniPoly::from_evals(&c),
            );

            // Get eval[3] for raf and rv_hw (which are degree 2, hence just output evals[0..=2]).
            // We need it to batch poly evaluations
            // For a quadratic poly: a + bx + cx²,
            // eval[3] = 3 · (eval[2] - eval[1]) + eval[0]
            let b_3 = (b[2] - b[1]).mul_u64(3) + b[0];
            let c_3 = (c[2] - c[1]).mul_u64(3) + c[0];
            raf.push(b_3);
            rv_hw.push(c_3);

            boolean
                .into_iter()
                .zip_eq(raf)
                .zip_eq(rv_hw)
                .map(|((a, b), c)| a + self.gamma * b + self.gamma_sqr * c)
                .collect()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        if round < self.lookups_vars {
            // Not yet to raf and rv_hw's variables
            self.booleanity.bind(r_j, round);

            let scaling_pow = self.lookups_vars - round - 1;
            let b_claim = self.raf.input_claim().mul_pow_2(scaling_pow);
            let c_claim = self.rv_hw.input_claim().mul_pow_2(scaling_pow);

            let (a, ..) = &self.unipolys;
            self.claims = (a.evaluate(&r_j), b_claim, c_claim);
        } else {
            self.booleanity.bind(r_j, round);
            self.raf.bind(r_j, round);
            self.rv_hw.bind(r_j, round);

            let (a, b, c) = &self.unipolys;
            self.claims = (a.evaluate(&r_j), b.evaluate(&r_j), c.evaluate(&r_j));
        }
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // Slice of the challenge used to bind raf and rv_hw polynomials; corresponds to `word_vars` last challenges
        let r_slice = r.split_at(self.lookups_vars).1;
        assert_eq!(r_slice.len(), self.words_vars);

        self.booleanity
            .expected_output_claim(opening_accumulator.clone(), r)
            + self.gamma
                * self
                    .raf
                    .expected_output_claim(opening_accumulator.clone(), r_slice)
            + self.gamma_sqr
                * self
                    .rv_hw
                    .expected_output_claim(opening_accumulator, r_slice)
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
        // Opening point has been normalized, and since it was bound in LowToHigh order for all sumcheck,
        // we need to take the first word_vars variables to get raf and rvhw opening point
        let r_slice = opening_point.split_at(self.words_vars).0;
        assert_eq!(r_slice.len(), self.words_vars);

        self.booleanity
            .cache_openings_prover(accumulator.clone(), opening_point.clone());
        self.raf
            .cache_openings_prover(accumulator.clone(), r_slice.clone());
        self.rv_hw.cache_openings_prover(accumulator, r_slice);
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_slice = opening_point.split_at(self.words_vars).0;
        assert_eq!(r_slice.len(), self.words_vars);

        self.booleanity
            .cache_openings_verifier(accumulator.clone(), opening_point.clone());
        self.raf
            .cache_openings_verifier(accumulator.clone(), r_slice.clone());
        self.rv_hw.cache_openings_verifier(accumulator, r_slice);
    }
}

// From the read addresses, computes the bound ra vector.
pub fn compute_ra_evals<F>(r: &[F], read_addresses: &[usize], K: usize) -> Vec<F>
where
    F: JoltField,
{
    let E = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = read_addresses.len().div_ceil(num_threads);
    let partial_results: Vec<Vec<F>> = read_addresses
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_ra = unsafe_allocate_zero_vec::<F>(K);
            let base_idx = chunk_idx * chunk_size;
            chunk.iter().enumerate().for_each(|(local_j, &k)| {
                let global_j = base_idx + local_j;
                local_ra[k] += E[global_j];
            });
            local_ra
        })
        .collect();
    let mut ra = unsafe_allocate_zero_vec::<F>(K);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}

#[cfg(test)]
mod tests {

    use crate::jolt::{
        pcs::OpeningId,
        precompiles::gather::{
            ExecutionProof,
            test::{TestInstance, VirtualOpening, random_gather, test_gather_sumcheck},
        },
    };

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
    use rand::rngs::StdRng;

    fn execution_instances<ProofTranscript, CS>(
        rng: &mut StdRng,
        prover_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        verifier_sm: &mut StateManager<'_, Fr, ProofTranscript, CS>,
        (num_lookups, num_words, word_dim): (usize, usize, usize),
        index: usize,
    ) -> (
        TestInstance,
        Box<dyn SumcheckInstance<Fr>>, // Prover instance
        Box<dyn SumcheckInstance<Fr>>, // Verifier instance
    )
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        // let mut test_instances: Vec<TestInstance> = Vec::new();
        // let mut prover_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();
        // let mut verifier_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();

        let (read_addresses, dictionnary, output) =
            random_gather(rng, num_lookups, num_words, word_dim);
        let test_instance = TestInstance::new(&read_addresses, &dictionnary, &output);

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
        virtual_openings.push(VirtualOpening::new(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.clone(),
            rv_claim_c,
        ));

        let rv_claim_a = MultilinearPolynomial::from(read_addresses).evaluate(r_x);
        virtual_openings.push(VirtualOpening::new(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            r_x.to_vec(),
            rv_claim_a,
        ));

        virtual_openings.iter().for_each(|opening| {
            opening.insert_in_prover_sm(prover_sm);
            opening.insert_in_verifier_sm(verifier_sm);
        });

        let prover_instance = ExecutionProof::gather_prover_instance(
            prover_sm,
            read_addresses_usize,
            dictionnary.clone(),
            word_dim,
            index,
        );
        let verifier_instance = ExecutionProof::gather_verifier_instance(
            verifier_sm,
            num_lookups,
            dictionnary.len(),
            word_dim,
            index,
        );
        (test_instance, prover_instance, verifier_instance)
    }

    #[test]
    fn test_execution_proof() {
        // Number of words to recover from the dictionnary
        const NUM_LOOKUPS: usize = 1 << 10;
        // Number of words in the dictionnary
        const NUM_WORDS: usize = 64;
        // Number of dimensions per word of dictionnary
        const WORD_DIM: usize = 1 << 3;

        let (instances, openings) = test_gather_sumcheck(
            execution_instances,
            (NUM_LOOKUPS, NUM_WORDS, WORD_DIM),
            123456,
            10,
        );

        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, dictionnary, _output) = instance.get();

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

            let ra_poly = MultilinearPolynomial::from(ra);

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRvHw,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, exp_claim, "Failed at index {i}");

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherBooleanity,
                ))
                .expect("GatherRa(index) should be set");

            let expected_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, expected_claim, "Failed at index {i}");

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRaf,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, exp_claim);

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::PrecompileB(i),
                    SumcheckId::GatherRvHw,
                ))
                .expect("PrecompileB(index) should be set");

            let exp_claim = MultilinearPolynomial::from(dictionnary).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }
}
