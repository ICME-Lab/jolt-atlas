use crate::onnx_proof::relation::{RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{op::Rsqrt, trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::rsqrt::*;

crate::impl_plan_op_prover!(Rsqrt);

impl<F: JoltField> RelationWitnesses<F> for RsqrtRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, output } = TraceView::layer_data(trace, &self.node);
        let [input] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let s_cubed = rsqrt_dividend(&self.node);
        // `quotient = ⌊S³ / x̂⌋` and `out²` can exceed i32 at higher scales, so both
        // are computed in i64; `quotient` is committed as u64. `div_remainder < x̂`
        // and `sqrt_remainder < 2·out+1` are small and stay i32.
        let quotient: Vec<u64> = input.iter().map(|&x| (s_cubed / x as i64) as u64).collect();
        let div_remainder: Vec<i32> = input.iter().map(|&x| (s_cubed % x as i64) as i32).collect();
        let sqrt_remainder: Vec<i32> = quotient
            .iter()
            .zip(output.iter())
            .map(|(&quotient, &out)| (quotient as i64 - (out as i64) * (out as i64)) as i32)
            .collect();
        #[cfg(test)]
        {
            use crate::onnx_proof::testing::probe::{probe_assert, probe_assert_eq};
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
            MultilinearPolynomial::from(div_remainder),
            MultilinearPolynomial::from(sqrt_remainder),
        ]
    }
}
