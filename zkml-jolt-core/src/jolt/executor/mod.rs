use crate::jolt::{
    dag::{stage::SumcheckStages, state_manager::StateManager},
    executor::{
        booleanity::BooleanitySumcheck, hamming_weight::HammingWeightSumcheck,
        instructions::LookupQuery, ra_virtual::RASumCheck, read_raf_checking::ReadRafSumcheck,
    },
    pcs::SumcheckId,
    sumcheck::SumcheckInstance,
    trace::{JoltONNXCycle, WORD_SIZE},
    witness::VirtualPolynomial,
};
use jolt_core::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial},
    transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
    zkvm::instruction_lookups::{D, K_CHUNK, LOG_K, LOG_K_CHUNK},
};
use rayon::prelude::*;

pub mod booleanity;
pub mod hamming_weight;
pub mod instructions;
pub mod ra_virtual;
pub mod read_raf_checking;

#[derive(Default)]
pub struct LookupsDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for LookupsDag
{
    fn stage3_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let (_, trace, _) = sm.get_prover_data();
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);
        let F = compute_ra_evals(trace, &eq_r_cycle);

        let read_raf = ReadRafSumcheck::new_prover(sm, eq_r_cycle.clone());
        let booleanity = BooleanitySumcheck::new_prover(sm, F.clone());
        let hamming_weight = HammingWeightSumcheck::new_prover(sm, F);

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage3_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_raf = ReadRafSumcheck::new_verifier(sm);
        let booleanity = BooleanitySumcheck::new_verifier(sm);
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage4_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let ra_virtual = RASumCheck::new_prover(LOG_K, sm);

        vec![Box::new(ra_virtual)]
    }

    fn stage4_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let ra_virtual = RASumCheck::new_verifier(LOG_K, sm);

        vec![Box::new(ra_virtual)]
    }
}

#[inline(always)]
fn compute_ra_evals<F: JoltField>(trace: &[JoltONNXCycle], eq_r_cycle: &[F]) -> [Vec<F>; D] {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: [Vec<F>; D] =
                std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK));
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<WORD_SIZE>::to_lookup_index(cycle);
                for i in (0..D).rev() {
                    let k = lookup_index % K_CHUNK as u64;
                    result[i][k as usize] += eq_r_cycle[j];
                    lookup_index >>= LOG_K_CHUNK;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK)),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}
