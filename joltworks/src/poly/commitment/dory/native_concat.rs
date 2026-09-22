//! Concatenate two equally shaped hidden tensors on a registered axis.
//! The output MLE selects between the two original input MLEs with one added
//! coordinate. Both signed input ranges are mandatory, including repeated inputs.

use super::{native_mul::Range, native_reduce::shape_bits};
use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        blindfold::{InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource},
        booleanity::BooleanitySumcheckVerifier,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_std::Zero;
use common::CommittedPoly;
use std::collections::BTreeMap;

fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
#[derive(Clone)]
pub(super) struct Concatenation {
    pub output_shape: Vec<usize>,
    pub log_input: usize,
    pub selector_bit: usize,
}
impl Concatenation {
    pub fn new(shape: &[usize], axis: usize) -> Result<Self, ProofVerifyError> {
        let log_input = shape_bits(shape)?;
        if axis >= shape.len() {
            return Err(invalid("Invalid concatenation axis"));
        }
        let mut output_shape = shape.to_vec();
        output_shape[axis] = output_shape[axis]
            .checked_mul(2)
            .ok_or_else(|| invalid("Concatenation shape overflow"))?;
        shape_bits(&output_shape)?;
        let selector_bit = shape[..axis].iter().map(|d| d.ilog2() as usize).sum();
        Ok(Self {
            output_shape,
            log_input,
            selector_bit,
        })
    }
    fn source(&self, i: usize) -> (usize, usize) {
        let remaining = self.log_input - self.selector_bit;
        let side = (i >> remaining) & 1;
        let index = ((i >> (remaining + 1)) << remaining) | (i & ((1 << remaining) - 1));
        (side, index)
    }
    fn input_point(&self, r: &[Fr]) -> Vec<Fr> {
        r.iter()
            .enumerate()
            .filter_map(|(i, v)| (i != self.selector_bit).then_some(*v))
            .collect()
    }
}
pub(super) struct ConcatWitness {
    pub output: Vec<i32>,
    pub range_values: Vec<Vec<u64>>,
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
}
impl ConcatWitness {
    pub fn new(inputs: [&[i32]; 2], layout: &Concatenation) -> Result<Self, ProofVerifyError> {
        if inputs.iter().any(|x| x.len() != 1 << layout.log_input) {
            return Err(invalid("Concatenation input length mismatch"));
        }
        let output = (0..1 << (layout.log_input + 1))
            .map(|i| {
                let (side, index) = layout.source(i);
                inputs[side][index]
            })
            .collect::<Vec<_>>();
        let range_values = inputs
            .iter()
            .map(|x| {
                x.iter()
                    .map(|v| (i64::from(*v) + (1i64 << 31)) as u64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut polynomials =
            BTreeMap::from([(tensor(2), MultilinearPolynomial::from(output.clone()))]);
        for (j, values) in range_values.iter().enumerate() {
            let r = Range::new(j, 32, 1 << 31);
            for d in 0..r.chunks() {
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(j, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        values
                            .iter()
                            .map(|v| Some(u16::from(r.digit(*v, d))))
                            .collect(),
                        1 << r.chunk,
                    )),
                );
            }
        }
        Ok(Self {
            output,
            range_values,
            polynomials,
        })
    }
}
pub(super) struct ConcatRegistration {
    pub tensors: [CommittedPoly; 3],
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
    pub layout: Concatenation,
}
impl ConcatRegistration {
    pub(super) fn ranges(&self) -> Vec<Range> {
        (0..2)
            .map(|j| {
                let mut r = Range::new(j, 32, 1 << 31);
                r.namespace = self.namespace + j;
                r.input = OpeningId::new(self.tensors[j], self.initial);
                r
            })
            .collect()
    }
    pub fn indicator_keys(&self) -> Vec<CommittedPoly> {
        self.ranges()
            .iter()
            .flat_map(|r| {
                (0..r.chunks()).map(move |d| CommittedPoly::NodeOutputRaD(r.namespace, d))
            })
            .collect()
    }
    fn params(&self, t: &mut Blake2bTranscript) -> ConcatParams {
        let output: Vec<Fr> = t.challenge_vector(self.layout.log_input + 1);
        let selector = output[self.layout.selector_bit];
        let input = self.layout.input_point(&output);
        ConcatParams {
            openings: self.tensors.map(|id| OpeningId::new(id, self.final_stage)),
            points: [input.clone(), input, output],
            coefficients: [-(Fr::from(1u64) - selector), -selector, Fr::from(1u64)],
        }
    }
    pub fn provers(
        &self,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>, ProofVerifyError> {
        if values.len() != 2 || values.iter().any(|v| v.len() != 1 << self.layout.log_input) {
            return Err(invalid("Invalid concatenation range witness"));
        }
        let params = self.params(t);
        let evaluations =
            std::array::from_fn(|j| polynomials[&self.tensors[j]].evaluate(&params.points[j]));
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        let mut result: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> =
            vec![Box::new(ConcatProver {
                params,
                evaluations,
            })];
        for range in self.ranges() {
            a.append_dense(
                t,
                range.input,
                r.clone(),
                polynomials[&self.tensors[range.tensor]].evaluate(&r),
            );
            let params = range.params(&r, t);
            result.push(Box::new(range.prover(&values[range.tensor], params)));
        }
        Ok(result)
    }
    pub fn verifiers(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let params = self.params(t);
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> =
            vec![Box::new(ConcatVerifier(params))];
        for range in self.ranges() {
            a.append_dense(t, range.input, r.clone());
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&r, t),
            )));
        }
        result
    }
}
#[derive(Clone)]
struct ConcatParams {
    openings: [OpeningId; 3],
    points: [Vec<Fr>; 3],
    coefficients: [Fr; 3],
}
impl SumcheckInstanceParams<Fr> for ConcatParams {
    fn degree(&self) -> usize {
        1
    }
    // The layout equality has no summed variables. Its mandatory range check
    // gives the joint stage positive length, including scalar-only graphs.
    fn num_rounds(&self) -> usize {
        0
    }
    fn input_claim(&self, _: &dyn OpeningAccumulator<Fr>) -> Fr {
        Fr::zero()
    }
    fn normalize_opening_point(&self, r: &[Fr]) -> OpeningPoint<BIG_ENDIAN, Fr> {
        r.to_vec().into()
    }
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::sum_of_products(vec![ProductTerm::single(ValueSource::Constant(0))])
    }
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<Fr>) -> Vec<Fr> {
        vec![]
    }
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::sum_of_products(
            (0..3)
                .map(|i| {
                    ProductTerm::scaled(
                        ValueSource::Challenge(i),
                        vec![ValueSource::Opening(self.openings[i])],
                    )
                })
                .collect(),
        ))
    }
    fn output_constraint_challenge_values(&self, _: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        self.coefficients.to_vec()
    }
}
#[derive(allocative::Allocative)]
struct ConcatProver {
    #[allocative(skip)]
    params: ConcatParams,
    #[allocative(skip)]
    evaluations: [Fr; 3],
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ConcatProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        unreachable!("Concatenation equality has no sumcheck rounds")
    }
    fn ingest_challenge(&mut self, _: <Fr as JoltField>::Challenge, _: usize) {
        unreachable!("Concatenation equality has no sumcheck rounds")
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..3 {
            a.append_dense(
                t,
                self.params.openings[i],
                self.params.points[i].clone(),
                self.evaluations[i],
            );
        }
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, f: &mut allocative::FlameGraphBuilder) {
        f.visit_root(self);
    }
}
struct ConcatVerifier(ConcatParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ConcatVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        _: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        (0..3)
            .map(|j| {
                self.0.coefficients[j] * a.get_committed_polynomial_opening(self.0.openings[j]).1
            })
            .sum()
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..3 {
            a.append_dense(t, self.0.openings[i], self.0.points[i].clone());
        }
    }
}
