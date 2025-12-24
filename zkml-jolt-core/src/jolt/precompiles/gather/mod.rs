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
    precompiles::gather::shout::{
        BooleanitySumcheck, HammingBooleanitySumcheck, HwSumcheck, RafSumcheck, ReadValueSumcheck,
    },
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

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
        let num_lookups_padded = num_lookups.next_power_of_two();
        let num_words_padded = dictionnary.len() / word_dim; // dictionnary is a num_words * word_dim matrix
        assert!(num_words_padded.is_power_of_two());

        // Recover challenge built during previous sumcheck
        let (r_c, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
        );

        let (r_x, r_y) = r_c.split_at(num_lookups_padded.log_2());
        assert_eq!(r_y.len(), word_dim.log_2());

        let F = compute_ra_evals(&r_x.r, &read_addresses, num_words_padded);

        let eq_r_y = EqPolynomial::evals(&r_y.r);
        // dictionnary, folded to a single column
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
        assert_eq!(dictionnary_folded.len(), num_words_padded);

        let execution_instance =
            ExecutionSumcheck::init_prover(sm, read_addresses, dictionnary_folded, F, index);

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
        dictionnary_length_padded: usize,
        word_dim_padded: usize,
        index: usize,
    ) -> Box<dyn SumcheckInstance<F>> {
        let num_words_padded = dictionnary_length_padded / word_dim_padded;
        assert!(num_words_padded.is_power_of_two());

        let execution_instance =
            ExecutionSumcheck::init_verifier(sm, num_lookups, num_words_padded, index);

        Box::new(execution_instance)
    }
}

pub struct ExecutionSumcheck<F: JoltField> {
    booleanity: BooleanitySumcheck<F>,
    hb: HammingBooleanitySumcheck<F>,
    raf: RafSumcheck<F>,
    rv: ReadValueSumcheck<F>,
    // folding challenge for different sumchecks
    claims: [F; 4],
    unipolys: [UniPoly<F>; 4],
    gamma_powers: Vec<F>,
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
        let num_lookups_padded = num_lookups.next_power_of_two();
        let num_words_padded = pp.b_dims[0];
        let word_dim_padded = pp.b_dims[1];

        let read_addresses = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|&x| x as usize)
            .take(num_lookups)
            .collect();
        let dictionnary = pp.extract_rv(final_memory_state, |m| &m.b_addr);
        let output = pp.extract_rv(final_memory_state, |m| &m.c_addr);

        let r_c: Vec<F> = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups_padded.log_2() + word_dim_padded.log_2());

        let (r_x, r_y) = r_c.split_at(num_lookups_padded.log_2());

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
        assert_eq!(r_y.len(), word_dim_padded.log_2());

        let F = compute_ra_evals(r_x, &read_addresses_usize, num_words_padded);

        let eq_r_y = EqPolynomial::evals(r_y);
        // dictionnary, folded to a single column
        let dictionnary_folded: Vec<F> = dictionnary
            .chunks(word_dim_padded)
            .map(|B_chunk| {
                B_chunk
                    .iter()
                    .zip_eq(eq_r_y.iter())
                    .map(|(&b, &e)| F::from_i64(b) * e)
                    .sum()
            })
            .collect();
        assert_eq!(dictionnary_folded.len(), num_words_padded);

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
        let hb = HammingBooleanitySumcheck::new_prover(sm, read_addresses.clone(), index);
        let raf = RafSumcheck::new_prover(sm, read_addresses.clone(), F.clone(), index);
        let rv = ReadValueSumcheck::new_prover(sm, read_addresses, dictionnary_folded, F, index);
        let claims = [
            booleanity.input_claim(),
            hb.input_claim(),
            raf.input_claim() * F::from_u32(lookups_vars.pow2() as u32),
            rv.input_claim() * F::from_u32(lookups_vars.pow2() as u32),
        ];
        let unipolys = [
            UniPoly::zero(),
            UniPoly::zero(),
            UniPoly::zero(),
            UniPoly::zero(),
        ];

        let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); 4];
        for i in 1..4 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            booleanity,
            hb,
            raf,
            rv,
            claims,
            unipolys,
            gamma_powers,
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
        let num_lookups_padded = num_lookups.next_power_of_two();
        let num_words_padded = pp.b_dims[0];
        let word_dim_padded = pp.b_dims[1];

        let r_c: Vec<F> = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(num_lookups_padded.log_2() + word_dim_padded.log_2());

        let (r_x, _r_y) = r_c.split_at(num_lookups_padded.log_2());

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

        Self::init_verifier(sm, num_lookups, num_words_padded, index)
    }

    pub fn init_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        num_lookups: usize,
        num_words: usize,
        index: usize,
    ) -> Self {
        let num_lookups_padded = num_lookups.next_power_of_two();
        let booleanity = BooleanitySumcheck::new_verifier(sm, num_lookups_padded, num_words, index);
        let hb = HammingBooleanitySumcheck::new_verifier(sm, num_lookups_padded, index);
        let raf = RafSumcheck::new_verifier(sm, num_lookups_padded, num_words, index);
        let rv = ReadValueSumcheck::new_verifier(sm, num_lookups_padded, num_words, index);

        let claims = [
            booleanity.input_claim(),
            hb.input_claim(),
            raf.input_claim() * F::from_u32(num_lookups_padded as u32),
            rv.input_claim() * F::from_u32(num_lookups_padded as u32),
        ];
        let unipolys = [
            UniPoly::zero(),
            UniPoly::zero(),
            UniPoly::zero(),
            UniPoly::zero(),
        ];

        let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); 4];
        for i in 1..4 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            booleanity,
            hb,
            raf,
            rv,
            claims,
            unipolys,
            gamma_powers,
            lookups_vars: num_lookups_padded.log_2(),
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
        self.gamma_powers[0] * self.booleanity.input_claim()
            + (self.gamma_powers[1] * self.hb.input_claim())
            + F::from_u32(self.lookups_vars.pow2() as u32)
                * (self.gamma_powers[2] * self.raf.input_claim()
                    + self.gamma_powers[3] * self.rv.input_claim())
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let inv2 = F::from_u32(2).inverse().unwrap();

        if round < self.lookups_vars {
            // Not yet to raf and rv's variables, only binding booleanity and hamming booleanity sumcheck
            let booleanity = self
                .booleanity
                .compute_prover_message(round, _previous_claim);
            let hb = self.hb.compute_prover_message(round, _previous_claim);

            // create and store Univariate Polynomial of current round
            let mut a_evals = booleanity.clone();
            a_evals.insert(1, self.claims[0] - a_evals[0]);
            let mut b_evals = hb.clone();
            b_evals.insert(1, self.claims[1] - b_evals[0]);

            // raf and rv are not yet getting bound, for now univariate polynomials are constant, divided by two at each round
            self.claims[2] *= inv2;
            self.claims[3] *= inv2;

            // Store booleanity sumcheck poly to later compute claim
            self.unipolys[0] = UniPoly::from_evals(&a_evals);
            self.unipolys[1] = UniPoly::from_evals(&b_evals);

            // raf and rv_hw polys are constant, hence we just add the gamma-weighted constants to the booleanity evaluations
            booleanity
                .iter()
                .zip(hb.iter())
                .map(|(&b, &h)| {
                    b * self.gamma_powers[0]
                        + h * self.gamma_powers[1]
                        + self.claims[2] * self.gamma_powers[2]
                        + self.claims[3] * self.gamma_powers[3]
                })
                .collect()
        } else {
            let boolean = self
                .booleanity
                .compute_prover_message(round, _previous_claim);
            let mut raf = self.raf.compute_prover_message(round, _previous_claim);
            let mut rv = self.rv.compute_prover_message(round, _previous_claim);
            let mut a = boolean.clone();
            let mut c = raf.clone();
            let mut d = rv.clone();

            // insert eval[1]
            a.insert(1, self.claims[0] - a[0]);
            c.insert(1, self.claims[2] - c[0]);
            d.insert(1, self.claims[3] - d[0]);

            // hb has been fully bound, univariate polynomials are the output claim, divided by two at each further round
            self.claims[1] *= inv2;

            self.unipolys = [
                UniPoly::from_evals(&a),
                UniPoly::zero(), // Not used anymore
                UniPoly::from_evals(&c),
                UniPoly::from_evals(&d),
            ];

            // Get eval[3] for raf and rv (which are degree 2, hence just output evals[0..=2]).
            // We need it to batch poly evaluations
            // For a quadratic poly: a + bx + cx²,
            // eval[3] = 3 · (eval[2] - eval[1]) + eval[0]
            let c_3 = (c[2] - c[1]).mul_u64(3) + c[0];
            let d_3 = (d[2] - d[1]).mul_u64(3) + d[0];
            raf.push(c_3);
            rv.push(d_3);

            boolean
                .into_iter()
                .zip_eq(raf)
                .zip_eq(rv)
                .map(|((a, c), d)| {
                    self.gamma_powers[0] * a
                        + self.gamma_powers[1] * self.claims[1]
                        + self.gamma_powers[2] * c
                        + self.gamma_powers[3] * d
                })
                .collect()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        if round < self.lookups_vars {
            // Not yet to raf and rv's variables
            self.booleanity.bind(r_j, round);
            self.hb.bind(r_j, round);

            let a = &self.unipolys[0];
            let b = &self.unipolys[1];
            self.claims = [
                a.evaluate(&r_j),
                b.evaluate(&r_j),
                self.claims[2],
                self.claims[3],
            ];
        } else {
            // Already totally bound hb
            self.booleanity.bind(r_j, round);
            self.raf.bind(r_j, round);
            self.rv.bind(r_j, round);

            let [a, _, c, d] = &self.unipolys;
            self.claims = [
                a.evaluate(&r_j),
                self.claims[1],
                c.evaluate(&r_j),
                d.evaluate(&r_j),
            ];
        }
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // Getting challenge slices for rv, raf and hb sumchecks
        let (r_lookups, r_words) = r.split_at(self.lookups_vars);
        assert_eq!(r_words.len(), self.words_vars);

        self.gamma_powers[0]
            * self
                .booleanity
                .expected_output_claim(opening_accumulator.clone(), r)
            + self.gamma_powers[1]
                * self
                    .hb
                    .expected_output_claim(opening_accumulator.clone(), r_lookups)
                * F::from_u32(self.words_vars.pow2() as u32)
                    .inverse()
                    .unwrap()
            + self.gamma_powers[2]
                * self
                    .raf
                    .expected_output_claim(opening_accumulator.clone(), r_words)
            + self.gamma_powers[3] * self.rv.expected_output_claim(opening_accumulator, r_words)
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
        // we need to take the first word_vars variables to get raf and rv opening point, and last lookups_vars for hb
        let (r_words, r_lookups) = opening_point.split_at(self.words_vars);
        assert_eq!(r_lookups.len(), self.lookups_vars);

        self.booleanity
            .cache_openings_prover(accumulator.clone(), opening_point);
        self.hb
            .cache_openings_prover(accumulator.clone(), r_lookups);
        self.raf
            .cache_openings_prover(accumulator.clone(), r_words.clone());
        self.rv.cache_openings_prover(accumulator, r_words);
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_words, r_lookups) = opening_point.split_at(self.words_vars);
        assert_eq!(r_lookups.len(), self.lookups_vars);

        self.booleanity
            .cache_openings_verifier(accumulator.clone(), opening_point);
        self.hb
            .cache_openings_verifier(accumulator.clone(), r_lookups);
        self.raf
            .cache_openings_verifier(accumulator.clone(), r_words.clone());
        self.rv.cache_openings_verifier(accumulator, r_words);
    }
}

// Wrapper to initialize HwSumcheck with preprocessing values
pub struct HammingWeightSumcheck<F: JoltField>(HwSumcheck<F>);

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn new_prover<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> HwSumcheck<F> {
        let final_memory_state = sm.get_val_final();
        let (pp, ..) = sm.get_prover_data();
        let pp = &pp.shared.precompiles.instances[index];

        let num_lookups = pp.a_dims[0];
        let num_words_padded = pp.b_dims[0];

        let read_addresses = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|&x| x as usize)
            .take(num_lookups)
            .collect();

        let (r_x, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::GatherHammingWeight(index),
            SumcheckId::GatherHB,
        );

        let F = compute_ra_evals(&r_x.r, &read_addresses_usize, num_words_padded);

        HwSumcheck::new_prover(sm, num_words_padded, F, index)
    }

    pub fn new_verifier<'a>(
        sm: &mut StateManager<'a, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> HwSumcheck<F> {
        let (pp, ..) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];

        let num_words_padded = pp.b_dims[0];

        HwSumcheck::new_verifier(sm, num_words_padded, index)
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
        (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
        index: usize,
    ) -> TestInstance<Fr>
    where
        ProofTranscript: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let test_io = random_gather(rng, max_num_lookups, max_num_words, max_word_dim);

        let (num_lookups, _, word_dim) = test_io.dims();
        let (read_addresses, dictionnary, output) = test_io.values();

        let read_addresses_usize: Vec<usize> = read_addresses
            .iter()
            .map(|e| e.to_u64().unwrap() as usize)
            .take(num_lookups)
            .collect();

        //------------------- Simulate challenges from previous stages of jolt-dag ----------------------
        // Reduce output matrix to a single evaluation

        let mut virtual_openings = Vec::new();
        let opening_c = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(output.clone()),
            None,
        );

        let (r_x, _) = opening_c.r.split_at(num_lookups.log_2());

        let opening_a = VirtualOpening::initial_claim(
            prover_sm,
            verifier_sm,
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            MultilinearPolynomial::from(read_addresses.clone()),
            Some(r_x.to_vec()),
        );

        virtual_openings.push(opening_c);
        virtual_openings.push(opening_a);

        let prover_instance = ExecutionProof::gather_prover_instance(
            prover_sm,
            read_addresses_usize,
            dictionnary.clone(),
            word_dim.next_power_of_two(),
            index,
        );
        let verifier_instance = ExecutionProof::gather_verifier_instance(
            verifier_sm,
            num_lookups,
            dictionnary.len(),
            word_dim.next_power_of_two(),
            index,
        );

        TestInstance::new(
            prover_instance,
            verifier_instance,
            virtual_openings,
            test_io,
        )
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
            1,
        );

        for (i, instance) in instances.into_iter().enumerate() {
            let (read_addresses, dictionnary, _) = instance.values();
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

            // Construct hamming weight claimed value, hw(i) = Sum_k( ra(i, k) ) for k in 0..num_words
            let hw: Vec<Fr> = ra
                .chunks(num_words.next_power_of_two())
                .map(|row| row.iter().sum())
                .collect();

            let ra_poly = MultilinearPolynomial::from(ra);

            // BooleanitySumcheck final claim
            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherBooleanity,
                ))
                .expect("GatherRa(index) should be set");

            let expected_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, expected_claim, "Failed at index {i}");

            // HammingBooleanitySumcheck final claim
            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherHammingWeight(i),
                    SumcheckId::GatherHB,
                ))
                .expect("GatherHammingBooleanity(index) should be set");

            let exp_claim = MultilinearPolynomial::from(hw).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);

            // RafSumcheck final claim
            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRaf,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, exp_claim);

            // ReadValueSumcheck final claims
            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::GatherRa(i),
                    SumcheckId::GatherRv,
                ))
                .expect("GatherRa(index) should be set");

            let exp_claim = ra_poly.evaluate(&r.r);
            assert_eq!(*claim, exp_claim, "Failed at index {i}");

            let (r, claim) = openings
                .get(&OpeningId::Virtual(
                    VirtualPolynomial::PrecompileB(i),
                    SumcheckId::PrecompileExecution,
                ))
                .expect("PrecompileB(index) should be set");

            let exp_claim = MultilinearPolynomial::from(dictionnary.clone()).evaluate(&r.r);
            assert_eq!(*claim, exp_claim);
        }
    }
}
