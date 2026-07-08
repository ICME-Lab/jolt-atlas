use crate::{
    onnx_proof::neural_teleport::{division::compute_division, utils::compute_ra_evals_direct},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::{
        consts::FOUR_PI_APPROX,
        trace::{LayerData, Trace},
    },
    node::ComputationNode,
};
use common::VirtualPoly;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningPoint, ProverOpeningAccumulator, BIG_ENDIAN},
    },
    transcripts::Transcript,
};

pub(super) struct TrigProverSetup<F: JoltField> {
    pub(super) input_onehot: MultilinearPolynomial<F>,
    pub(super) output_claim: F,
    pub(super) remainder_claim: F,
}

pub(super) fn initialize_trig_prover<F: JoltField>(
    trace: &Trace,
    computation_node: &ComputationNode,
    r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
    log_table_size: usize,
) -> TrigProverSetup<F> {
    let LayerData { operands, output } = Trace::layer_data(trace, computation_node);
    let input = operands[0];
    let (_quotient_tensor, remainder_tensor) = compute_division(input, FOUR_PI_APPROX);

    assert!(remainder_tensor
        .iter()
        .all(|&x| (0..FOUR_PI_APPROX).contains(&x)));

    let input_onehot = MultilinearPolynomial::from(compute_ra_evals_direct(
        &r_node_output.r,
        &remainder_tensor,
        1 << log_table_size,
    ));

    let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output.r);
    // Special case where we add a new opening for the node output at node's own index.
    // This is due to the fact that `remainder` poly claim is used as input claim for trig lookup
    // sumcheck, together with a node output claim. Both claims are derived from a different
    // opening point, so we need to derive a new claim for one of (`output`, `remainder`).
    // We chose `output` to prevent us from having to use n-to-1 reductions on the `remainder`
    // and rather only implement it on NodeOutput. Making further work for Issue#138 easier.
    let mut provider = AccOpeningAccessor::new(accumulator, computation_node)
        .into_provider(transcript, r_node_output.clone());
    provider.append_nodeio(Target::Current, output_claim);

    TrigProverSetup {
        input_onehot,
        output_claim,
        remainder_claim: provider.get_advice(VirtualPoly::TeleportRemainder).1,
    }
}

#[cfg(test)]
pub(super) fn assert_trig_lookup_claim<F: JoltField>(
    input_onehot: &MultilinearPolynomial<F>,
    table: &MultilinearPolynomial<F>,
    gamma: F,
    output_claim: F,
    remainder_claim: F,
) {
    let claim = (0..input_onehot.len())
        .map(|i| {
            let a = input_onehot.get_bound_coeff(i);
            let b = table.get_bound_coeff(i);
            let int = F::from_u32(i as u32);
            a * (b + gamma * int)
        })
        .sum();
    assert_eq!(output_claim + gamma * remainder_claim, claim);
}
