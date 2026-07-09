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
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Add,
};
use common::{CommittedPoly, VirtualPoly};
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
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
///
/// The sumcheck structs below ([`AddParams`]/[`AddProver`]/[`AddVerifier`]) are
/// retained for the ZK (BlindFold) path in `onnx_proof::zk`, which still proves
/// the un-clamped addition via a batched sumcheck; folding the clamp into that
/// path is tracked separately.
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
        let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Add operation")
        };
        let left_claim =
            MultilinearPolynomial::from(left.padded_next_power_of_two()).evaluate(&opening_point.r);
        let right_claim = MultilinearPolynomial::from(right.padded_next_power_of_two())
            .evaluate(&opening_point.r);
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

/// Shared parameter block for the element-wise addition sumcheck proof.
#[derive(Clone)]
pub struct AddParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> AddParams<F> {
    /// Creates new params by reading the current output opening from the accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AddParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_node_output_opening(self.computation_node.idx)
            .1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        use joltworks::utils::math::Math;
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    // output = eq_eval * (left + right) = eq_eval * left + eq_eval * right
    // Two terms, each with Challenge(0) = eq_eval scaling one opening.
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let op_builder =
            crate::utils::opening_access::OpeningIdBuilder::new(&self.computation_node);
        let left_id = op_builder.nodeio(Target::Input(0));
        let right_id = op_builder.nodeio(Target::Input(1));

        let terms = vec![
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(left_id)],
            ),
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(right_id)],
            ),
        ];
        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_node_output_prime: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_node_output_prime);
        vec![eq_eval]
    }
}

/// Prover state for element-wise addition sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomials needed to generate
/// sumcheck messages proving that output[i] = left[i] + right[i] for all i.
pub struct AddProver<F: JoltField> {
    params: AddParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> AddProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "AddProver::initialize")]
    pub fn initialize(trace: &Trace, params: AddParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Add operation")
        };
        let left_operand = MultilinearPolynomial::from(left_operand.padded_next_power_of_two());
        let right_operand = MultilinearPolynomial::from(right_operand.padded_next_power_of_two());
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for AddProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            right_operand,
            ..
        } = self;
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let ro0 = right_operand.get_bound_coeff(2 * g);
            [lo0 + ro0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0), self.left_operand.final_claim());
        provider.append_nodeio(Target::Input(1), self.right_operand.final_claim());
    }
}

/// Verifier for element-wise addition sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// addition operation output.
pub struct AddVerifier<F: JoltField> {
    params: AddParams<F>,
}

impl<F: JoltField> AddVerifier<F> {
    /// Create a new verifier for the addition operation.
    #[tracing::instrument(skip_all, name = "AddVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = AddParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for AddVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output = &self.params.r_node_output.r;
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(r_node_output, &r_node_output_prime);

        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let left_operand_claim = accessor.get_nodeio(Target::Input(0)).1;
        let right_operand_claim = accessor.get_nodeio(Target::Input(1)).1;
        eq_eval * (left_operand_claim + right_operand_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0));
        provider.append_nodeio(Target::Input(1));
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn add_model(rng: &mut StdRng, T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::random_small(rng, &[T]));
        let res = b.add(i, c);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_add() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = add_model(&mut rng, T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_add_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = add_model(&mut rng, t);
        unit_test_op(model, &[input]);
    }

    /// Small tensors: `T = 1` takes the scalar direct-check path; `T` ∈ {2, 4, 8}
    /// exercise the clamp lookup's few-cycle-round one-hot sumchecks
    /// (`RaPolynomial` `Round2`/`Round3` final claims).
    #[test]
    fn test_add_small_tensors() {
        for t in [1usize, 2, 4, 8] {
            let input = Tensor::<i32>::new(Some(&vec![3; t]), &[t]).unwrap();
            let mut b = ModelBuilder::new();
            let i = b.input(vec![t]);
            let c = b.constant(Tensor::<i32>::new(Some(&vec![2; t]), &[t]).unwrap());
            let res = b.add(i, c);
            b.mark_output(res);
            unit_test_op(b.build(), &[input]);
        }
    }

    /// Scalar (`1×1`) saturation: operands open in the clear, so the verifier
    /// recovers them and checks the clamp directly. Covers +/- saturation and a
    /// non-saturating case.
    #[test]
    fn test_add_scalar_saturates() {
        for (a, c_val) in [(i32::MAX, i32::MAX), (i32::MIN, i32::MIN), (i32::MAX, -5)] {
            let input = Tensor::<i32>::new(Some(&[a]), &[1]).unwrap();
            let mut bld = ModelBuilder::new();
            let i = bld.input(vec![1]);
            let c = bld.constant(Tensor::<i32>::new(Some(&[c_val]), &[1]).unwrap());
            let res = bld.add(i, c);
            bld.mark_output(res);
            unit_test_op(bld.build(), &[input]);
        }
    }

    /// Inputs chosen so that `left + right` saturates at **both** `i32::MAX`
    /// (even indices) and `i32::MIN` (odd indices), exercising the clamp lookup.
    #[test]
    fn test_add_saturating_overflow() {
        let t = 1 << 10;
        let input_data: Vec<i32> = (0..t)
            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
            .collect();
        let const_data: Vec<i32> = (0..t)
            .map(|i| if i % 2 == 0 { 1000 } else { -1000 })
            .collect();
        let input = Tensor::<i32>::new(Some(&input_data), &[t]).unwrap();
        let mut b = ModelBuilder::new();
        let i = b.input(vec![t]);
        let c = b.constant(Tensor::<i32>::new(Some(&const_data), &[t]).unwrap());
        let res = b.add(i, c);
        b.mark_output(res);
        unit_test_op(b.build(), &[input]);
    }
}
