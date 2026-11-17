//! The prover's side of the log-partition softmax
//! ([`jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis`]): the openings, the sums
//! relation's witnesses, the table read and the clamp lookup, all from the trace's
//! [`SoftmaxWitness`].
use crate::onnx_proof::op_lookups::{LookupWitness, OpLookupProve};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::IntoProverProvider;
use atlas_sumcheck::relation::prover::DynRelationWitness;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::*;

use crate::{
    onnx_proof::{
        op_lookups::OpLookupProvider,
        ops::softmax_last_axis::significance_clamp::SoftmaxSignificanceClampOperands,
    },
    utils::{compute_lookup_indices, opening_access::AccOpeningAccessor},
};
use atlas_lookup_tables::clamp::SoftmaxClampTable;
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
use atlas_transcript::Transcript;
use common::{consts::XLEN, CommittedPoly, VirtualPoly};

pub mod significance_clamp;

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

/// A softmax lookup whose read addresses the one-hot checks consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SoftmaxAddresses {
    /// The table read.
    Read,
    /// The clamp lookup.
    Clamp,
}

/// Append a softmax opening with the prover's claims from the trace.
pub fn prove_open<F: JoltField, T: Transcript>(
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

/// A softmax relation's provers, with the read addresses each lookup leaves for its one-hot
/// checks.
pub type StageProvers<F, T> = (
    Vec<Box<dyn DynRelationWitness<F, T>>>,
    Vec<(SoftmaxAddresses, Vec<usize>)>,
);

/// The provers of a softmax relation, set up against the prover's accumulator and transcript.
/// `addresses` hands back the read addresses a lookup of this node left.
pub fn prove_relation<F: JoltField, T: Transcript>(
    relation: SoftmaxRelation,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    addresses_left: &mut impl FnMut(SoftmaxAddresses) -> Option<Vec<usize>>,
) -> StageProvers<F, T> {
    let taken = |left: &mut dyn FnMut(SoftmaxAddresses) -> Option<Vec<usize>>,
                 key: SoftmaxAddresses| {
        left(key).unwrap_or_else(|| {
            panic!("softmax: the {key:?} addresses were not left by their lookup")
        })
    };
    match relation {
        SoftmaxRelation::Read(shape) => {
            let w = witness(&shape, trace);
            let read = tables();
            let indices = table_addresses(&shape.node, w);
            let prover = shout::prover::read_raf_prover(
                &ReadProvider { shape },
                &indices,
                &[&read.tables.out, &read.tables.sum, &read.sum_prev],
                &*accumulator,
                transcript,
            );
            (vec![prover], vec![(SoftmaxAddresses::Read, indices)])
        }
        SoftmaxRelation::Clamp(shape) => {
            let w = witness(&shape, trace);
            let u = addresses(&shape.node, w);
            let indices: Vec<usize> = compute_lookup_indices(&u)
                .iter()
                .map(|&x| x.into())
                .collect();
            let u = u.map(|v| v as i64);
            let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                OpLookupProvider::new(shape.node.clone());
            provider.append_witness_claim_for(&u, accumulator, transcript);
            let prover = provider.read_raf_prover_from_bits::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                provider.helper.lookup_bits(&u),
                accumulator,
                transcript,
            );
            (
                vec![Box::new(prover)],
                vec![(SoftmaxAddresses::Clamp, indices)],
            )
        }
        SoftmaxRelation::ReadOneHot(shape) => {
            let indices = taken(addresses_left, SoftmaxAddresses::Read);
            (
                shout::prover::ra_onehot_provers(
                    &ReadRaEncoding {
                        node_idx: shape.idx(),
                    },
                    &indices,
                    &*accumulator,
                    transcript,
                )
                .into(),
                Vec::new(),
            )
        }
        SoftmaxRelation::ClampOneHot(shape) => {
            let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                OpLookupProvider::new(shape.node.clone());
            let indices = taken(addresses_left, SoftmaxAddresses::Clamp);
            (
                shout::prover::ra_onehot_provers(
                    &provider.encoding(),
                    &indices,
                    &*accumulator,
                    transcript,
                )
                .into(),
                Vec::new(),
            )
        }
    }
}

/// The claim of an [`address_link`]: the operand at the reads' point.
pub fn address_link_claims<F: JoltField>(
    node: &crate::node_view::ComputationNode,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    vec![TraceView::operand_tensors(trace, node)[0].evaluate_mle(point)]
}
