use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::rsqrt::*;

/// The witnesses of a [`reciprocal_sqrt`]: the input, the division quotient, the output and
/// the two remainders.
pub fn reciprocal_sqrt_witnesses<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, output } = TraceView::layer_data(trace, node);
    let [input] = operands[..] else {
        panic!("Expected one operand for Rsqrt operation")
    };
    // `quotient = ⌊S³ / x̂⌋` can exceed i32 at higher scales and is committed as u64;
    // `div_remainder < x̂` and `sqrt_remainder < 2·out+1` are small and stay i32.
    let advice = TraceView::rsqrt_advice(trace, node);
    let quotient: Vec<u64> = advice.quotient.iter().map(|&q| q as u64).collect();
    let (div_remainder, sqrt_remainder) =
        (advice.div_remainder.data(), advice.sqrt_remainder.data());
    #[cfg(test)]
    {
        use crate::onnx_proof::testing::probe::{probe_assert, probe_assert_eq};
        let s_cubed = rsqrt_dividend(node);
        for (i, (&x, &q)) in input.iter().zip(&quotient).enumerate() {
            let (x, q, out) = (x as i128, q as i128, output.data()[i] as i128);
            let (div_remainder, sqrt_remainder) =
                (div_remainder[i] as i128, sqrt_remainder[i] as i128);
            // range checks: div_remainder < x, sqrt_remainder <= 2*output
            probe_assert!(div_remainder >= 0 && div_remainder < x);
            probe_assert!(sqrt_remainder >= 0 && sqrt_remainder <= 2 * out);
            probe_assert_eq!(x * q + div_remainder, s_cubed as i128);
            probe_assert_eq!(out * out + sqrt_remainder, q);
        }
    }
    vec![
        MultilinearPolynomial::from(input.clone()),
        MultilinearPolynomial::from(quotient),
        MultilinearPolynomial::from(output.clone()),
        MultilinearPolynomial::from(div_remainder.to_vec()),
        MultilinearPolynomial::from(sqrt_remainder.to_vec()),
    ]
}
