use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{
        clamp_lookups::{
            clamp_committed_polys, is_scalar, prove_clamp_lookup, verify_clamp_lookup,
            verify_scalar_clamp,
        },
        ops::OperatorProofTrait,
        ProofId, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Add, trace::LayerData};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::EvaluateMle,
        opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Proof for saturating element-wise addition: `output = SatClamp(left + right)`.
///
/// The accumulation is widened to `i64` (`acc = left + right`) and then clamped
/// to `i32`, so `Add` is no longer a pure linear identity. The proof has three
/// parts (mirroring [`super::relu`], but over a 64-bit clamp address):
///
/// 1. **Clamp lookup** ([`ClampLookupProvider`], `ProofType::Execution`): a
///    prefix-suffix read-raf sumcheck proving `output(r) = SatClamp(acc(r))` and
///    tying the accumulation MLE ([`VirtualPoly::ClampAcc`], the `raf`) to the
///    one-hot read-address poly ([`VirtualPoly::ClampRa`]).
/// 2. **One-hot checks** ([`ClampEncoding`], `ProofType::RaOneHotChecks`):
///    booleanity + hamming-weight + ra-virtualization of `ClampRa` against its
///    committed decomposition ([`CommittedPoly::ClampRaD`]).
/// 3. **Operand tie** (no sumcheck): the prover opens `left`/`right` at the same
///    output point `r`, and the verifier checks `left(r) + right(r) == acc(r)`.
///    `acc(r)` is the lookup's `raf` claim, already tied to `ClampRa` in (1).
///
/// The accumulation is re-executed from the operands at proving time
/// ([`clamp_intermediate`](crate::onnx_proof::clamp_lookups)); it is not stored
/// in the trace.
impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Add {
    #[tracing::instrument(skip_all, name = "Add::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // Output opening point `r` (shared by the lookup `r_cycle` and the tie).
        let (opening_point, _claim) =
            AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();

        // Non-scalar: prove `output = SatClamp(acc)` via the clamp lookup + one-hot.
        // Scalar: skipped — the verifier recomputes the clamp from the operands.
        let results = if is_scalar(node) {
            vec![]
        } else {
            prove_clamp_lookup(node, prover, None)
        };

        // Operand tie: open `left(r)`, `right(r)`. The verifier ties them to
        // `acc` (non-scalar) or recomputes `SatClamp(left + right)` (scalar).
        let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Add operation")
        };
        let left_claim = left.evaluate_mle(&opening_point.r);
        let right_claim = right.evaluate_mle(&opening_point.r);
        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
            .into_provider(&mut prover.transcript, opening_point);
        provider.append_nodeio(Target::Input(0), left_claim);
        provider.append_nodeio(Target::Input(1), right_claim);

        results
    }

    #[tracing::instrument(skip_all, name = "Add::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Non-scalar: verify `output = SatClamp(acc)` (lookup + one-hot).
        if !is_scalar(node) {
            verify_clamp_lookup(node, verifier)?;
        }

        // Read `left(r)`, `right(r)`.
        let (opening_point, output_claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, node)
            .into_provider(&mut verifier.transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
        provider.append_nodeio(Target::Input(1));
        let left_claim = provider.get_nodeio(Target::Input(0)).1;
        let right_claim = provider.get_nodeio(Target::Input(1)).1;
        drop(provider);

        if is_scalar(node) {
            // Operands open in the clear: `output == SatClamp(left + right)`.
            verify_scalar_clamp(left_claim + right_claim, output_claim, "Add")
        } else {
            // Tie `left(r) + right(r) == acc(r)`; `acc` is tied to `ClampRa` by
            // the lookup, and `output = SatClamp(acc)` was proven above.
            let acc_id = OpeningId::new(
                VirtualPoly::ClampAcc(node.idx),
                SumcheckId::NodeExecution(node.idx),
            );
            let acc_claim = verifier
                .accumulator
                .get_virtual_polynomial_opening(acc_id)
                .1;
            if left_claim + right_claim != acc_claim {
                return Err(ProofVerifyError::InvalidOpeningProof(
                    "Add: left + right must equal the i64 accumulation (pre-clamp)".to_string(),
                ));
            }
            Ok(())
        }
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamp_committed_polys(node)
    }
}
