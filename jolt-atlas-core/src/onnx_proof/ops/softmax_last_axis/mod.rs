//! The prover's side of the log-partition softmax
//! ([`jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis`]): the openings, the sums
//! relation's witnesses and the table read, all from the trace's
//! [`SoftmaxWitness`].
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::{AccOpeningAccessor, IntoProverProvider};
use atlas_sumcheck::relation::prover::DynRelationWitness;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::*;

use atlas_sumcheck::opening_proof::prover::ProverOpeningAccumulator;
use atlas_trace::{SoftmaxWitness, Tensor, Trace};

use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear::{EvaluateMle, MultilinearPolynomial, PolynomialEvaluation},
        point::OpeningPoint,
    },
};
use atlas_sumcheck::shout;
use atlas_transcript::ProvingTranscript;
use common::{CommittedPoly, VirtualPoly};

fn witness<'a>(shape: &SoftmaxShape, trace: &'a Trace) -> &'a SoftmaxWitness {
    trace
        .softmax_witness(shape.idx())
        .unwrap_or_else(|| panic!("softmax node {} has no trace witness", shape.idx()))
}

fn tables() -> &'static ReadTables {
    read_tables().expect("the softmax tables are the clamp table's width at the model scale")
}

/// The addresses `u = c − x` over the node's padded cells.
pub(crate) fn addresses(
    node: &crate::node_view::ComputationNode,
    w: &SoftmaxWitness,
) -> Tensor<i32> {
    Tensor::from_slice(&w.address, &node.output_dims)
        .expect("the softmax witness has one address per cell")
        .padded_pow2()
}

/// The table addresses `ũ = min(max(u, 0), top)` over the node's padded cells.
pub(crate) fn table_addresses(
    node: &crate::node_view::ComputationNode,
    w: &SoftmaxWitness,
) -> Vec<usize> {
    let top = tables().tables.top();
    addresses(node, w)
        .data()
        .iter()
        .map(|&u| (u as i64).clamp(0, top) as usize)
        .collect()
}

/// The values of a per-row vector as field elements, padded to the node's rows.
fn row_values<F: JoltField>(shape: &SoftmaxShape, values: impl Iterator<Item = i64>) -> Vec<F> {
    let mut values: Vec<F> = values.map(F::from_i64).collect();
    values.resize(shape.F_N[0], F::zero());
    values
}

/// Append a softmax opening with the prover's claims from the trace.
pub fn prove_open<F: JoltField, T: ProvingTranscript>(
    open: &SoftmaxOpen,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) {
    let expect = "the prover holds every claim it reads";
    match open {
        SoftmaxOpen::Gaps(shape) => {
            let w = witness(shape, trace);
            let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let r0_row = OpeningPoint::new(r0.r[..shape.log_f()].to_vec());
            let gap =
                MultilinearPolynomial::from(row_values::<F>(shape, w.sum_gap.iter().copied()))
                    .evaluate(&r0_row.r);
            let prev_gap =
                MultilinearPolynomial::from(row_values::<F>(shape, w.prev_sum_gap.iter().copied()))
                    .evaluate(&r0_row.r);
            let mut provider = accessor.into_provider(transcript, r0_row);
            provider.append_advice(VirtualPoly::SoftmaxSumGap, gap);
            provider.append_advice(VirtualPoly::SoftmaxPrevSumGap, prev_gap);
        }
        SoftmaxOpen::Addresses(shape) => {
            let w = witness(shape, trace);
            let r1 = AccOpeningAccessor::new(&*accumulator, &shape.node)
                .get_advice(VirtualPoly::SoftmaxRead)
                .expect(expect)
                .0;
            let r1_row = OpeningPoint::new(r1.r[..shape.log_f()].to_vec());
            let shift = MultilinearPolynomial::from(row_values::<F>(
                shape,
                w.shift.iter().map(|&c| i64::from(c)),
            ))
            .evaluate(&r1_row.r);
            AccOpeningAccessor::new(&mut *accumulator, &shape.node)
                .into_provider(transcript, r1_row)
                .append_advice(CommittedPoly::SoftmaxShift, shift);
            let clamped: Vec<i32> = table_addresses(&shape.node, w)
                .into_iter()
                .map(|a| a as i32)
                .collect();
            let clamped = clamped.evaluate_mle(&r1.r);
            AccOpeningAccessor::new(accumulator, &shape.node)
                .into_provider(transcript, r1)
                .append_advice(VirtualPoly::SoftmaxClampedAddr, clamped);
        }
    }
}

/// The witnesses of a [`sums`] relation: the reads `v`, `t`, `t⁻` over the node's padded
/// cells.
pub fn sums_witnesses<F: JoltField>(
    node: &crate::node_view::ComputationNode,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let w = trace
        .softmax_witness(node.idx)
        .unwrap_or_else(|| panic!("softmax node {} has no trace witness", node.idx));
    let read = tables();
    let addresses = table_addresses(node, w);
    [&read.tables.out, &read.tables.sum, &read.sum_prev]
        .into_iter()
        .map(|table| {
            MultilinearPolynomial::from(addresses.iter().map(|&a| table[a]).collect::<Vec<i32>>())
        })
        .collect()
}

/// The prover of the table read at the reads' point, the mirror of [`read_verifier`], with
/// the table addresses it reads, for their one-hot checks.
pub fn read_prover<F: JoltField, T: ProvingTranscript>(
    shape: &SoftmaxShape,
    trace: &Trace,
    accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> (Box<dyn DynRelationWitness<F, T>>, Vec<usize>) {
    let w = witness(shape, trace);
    let read = tables();
    let indices = table_addresses(&shape.node, w);
    let prover = shout::prover::read_raf_prover(
        &ReadProvider {
            shape: shape.clone(),
        },
        &indices,
        &[&read.tables.out, &read.tables.sum, &read.sum_prev],
        accumulator,
        transcript,
    );
    (prover, indices)
}

/// The claim of an [`address_link`]: the operand at the reads' point.
pub fn address_link_claims<F: JoltField>(
    node: &crate::node_view::ComputationNode,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    vec![TraceView::operand_tensors(trace, node)[0].evaluate_mle(point)]
}
