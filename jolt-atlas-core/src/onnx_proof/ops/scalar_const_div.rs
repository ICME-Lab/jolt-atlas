use crate::{
    onnx_proof::{
        ops::OperatorProofTrait,
        range_checking::{
            range_check_operands::{RangeCheckOperands, ScalarConstDivRangeCheckOperands},
            RangeCheckProvider,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::{
        adjusted_remainder,
        opening_access::{AccOpeningAccessor, Target},
    },
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, ScalarConstDiv},
};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ScalarConstDiv {
    #[tracing::instrument(skip_all, name = "ScalarConstDiv::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // Plumbing: cache the remainder's claim.
        cache_remainder_prover(node, prover);

        // Range check `0 <= R < b`.
        let rangecheck_provider = RangeCheckProvider::<ScalarConstDivRangeCheckOperands>::new(node);
        let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        let (rangecheck_proof, _) = Sumcheck::prove(
            &mut rangecheck_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        // The range check's read-address one-hot correctness checks.
        let rc_operands = RangeCheckOperands::<ScalarConstDivRangeCheckOperands>::new(node);
        let encoding = rc_operands.get_encoding(node);
        let [ra_sumcheck, hw_sumcheck, bool_sumcheck] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_sumcheck, hw_sumcheck, bool_sumcheck];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        // Plumbing: derive `a`'s input claim from `q` (this node's own output) and `R`.
        cache_input_claim_prover(node, prover);

        vec![
            (ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof),
            (
                ProofId(node.idx, ProofType::RaOneHotChecks),
                ra_one_hot_proof,
            ),
        ]
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDiv::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Plumbing: cache the remainder's claim.
        cache_remainder_verifier(node, verifier);

        // Range check `0 <= R < b`.
        let rangecheck_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RangeCheck))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let rangecheck_provider = RangeCheckProvider::<ScalarConstDivRangeCheckOperands>::new(node);
        let rangecheck_verifier = rangecheck_provider
            .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                &mut verifier.accumulator,
                &mut verifier.transcript,
            );
        Sumcheck::verify(
            rangecheck_proof,
            &rangecheck_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // The range check's read-address one-hot correctness checks.
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let rc_operands = RangeCheckOperands::<ScalarConstDivRangeCheckOperands>::new(node);
        let encoding = rc_operands.get_encoding(node);
        let [ra_sumcheck, hw_sumcheck, bool_sumcheck] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&*ra_sumcheck, &*hw_sumcheck, &*bool_sumcheck],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Plumbing: derive and verify `a`'s input claim from `q` and `R`.
        verify_input_claim(node, verifier)?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let rc_operands = RangeCheckOperands::<ScalarConstDivRangeCheckOperands>::new(node);
        let d = rc_operands
            .get_encoding(node)
            .one_hot_params()
            .instruction_d;
        (0..d)
            .map(|i| CommittedPoly::ScalarConstDivRangeCheckRaD(node.idx, i))
            .collect()
    }
}

/// Caches the remainder's claim (`VirtualPoly::ScalarConstDivRemainder`) at the node's
/// reduced output-opening point, by directly evaluating the honest remainder tensor —
/// no sumcheck needed, `R` is constrained only via [`cache_input_claim_prover`] and the
/// range check.
fn cache_remainder_prover<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) {
    let Operator::ScalarConstDiv(op) = &node.operator else {
        panic!("Expected ScalarConstDiv operator at node {}", node.idx);
    };
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
    let [left_operand] = operands[..] else {
        panic!("Expected one operand for ScalarConstDiv operation")
    };
    let remainder = left_operand.map(|x| adjusted_remainder(x, op.divisor));

    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let eval = MultilinearPolynomial::from(remainder).evaluate(&r_node_output.r);
    accessor
        .into_provider(&mut prover.transcript, r_node_output)
        .append_advice(VirtualPoly::ScalarConstDivRemainder, eval);
}

/// Verifier counterpart of [`cache_remainder_prover`].
fn cache_remainder_verifier<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) {
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    accessor
        .into_provider(&mut verifier.transcript, r_node_output)
        .append_advice(VirtualPoly::ScalarConstDivRemainder);
}

/// Derives the node's input claim `a = b·q + R` from `q` (this node's own reduced output
/// claim) and the remainder claim `R`, caching it as `Target::Input(0)`. Holds identically
/// as multilinear polynomials, so this is pure field arithmetic — no sumcheck needed. Must
/// run after [`cache_remainder_prover`].
fn cache_input_claim_prover<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) {
    let Operator::ScalarConstDiv(op) = &node.operator else {
        panic!("Expected ScalarConstDiv operator at node {}", node.idx);
    };
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, q_claim) = accessor.get_reduced_opening();
    let r_claim = accessor.get_advice(VirtualPoly::ScalarConstDivRemainder).1;
    let a_claim = F::from_i32(op.divisor) * q_claim + r_claim;

    accessor
        .into_provider(&mut prover.transcript, r_node_output)
        .append_nodeio(Target::Input(0), a_claim);
}

/// Verifier counterpart of [`cache_input_claim_prover`]: independently recomputes `b·q + R`
/// and checks it against the prover's claimed `Target::Input(0)`.
fn verify_input_claim<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let Operator::ScalarConstDiv(op) = &node.operator else {
        panic!("Expected ScalarConstDiv operator at node {}", node.idx);
    };
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let (r_node_output, q_claim) = accessor.get_reduced_opening();

    let mut provider = accessor.into_provider(&mut verifier.transcript, r_node_output);
    provider.append_nodeio(Target::Input(0));
    let r_claim = provider.get_advice(VirtualPoly::ScalarConstDivRemainder).1;
    let expected = F::from_i32(op.divisor) * q_claim + r_claim;
    let input_claim = provider.get_nodeio(Target::Input(0)).1;

    if input_claim != expected {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "ScalarConstDiv input claim does not match b*q + R derived from the quotient and \
             remainder claims"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn scalar_const_div_model(T: usize, divisor: i32) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.scalar_const_div(i, divisor);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_scalar_const_div() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = scalar_const_div_model(T, 128);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_scalar_const_div_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = scalar_const_div_model(t, 128);
        unit_test_op(model, &[input]);
    }
}
