//! Exact floor division and nonnegative remainder by a registered divisor.
//! Signed x and q, unsigned r and g satisfy x=D*q+r and r+g=D-1.
use super::native_mul::Range;
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
const N: usize = 4;
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
pub(super) fn ranges(divisor: u32) -> Vec<Range> {
    let bits = divisor.next_power_of_two().ilog2() as usize;
    vec![
        Range::new(0, 32, 1 << 31),
        Range::new(1, 32, 1 << 31),
        Range::new(2, bits, 0),
        Range::new(3, bits, 0),
    ]
}
pub(super) struct DivisionWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub output: Vec<i32>,
}
impl DivisionWitness {
    pub fn new(input: &[i32], divisor: u32, remainder: bool) -> Result<Self, ProofVerifyError> {
        if input.is_empty() || !input.len().is_power_of_two() || !(2..=1 << 30).contains(&divisor) {
            return Err(invalid("Unsupported integer division shape or divisor"));
        }
        let mut columns: [Vec<i32>; N] = std::array::from_fn(|_| Vec::with_capacity(input.len()));
        for x in input {
            let q = x.div_euclid(divisor as i32);
            let r = x.rem_euclid(divisor as i32);
            for (i, v) in [*x, q, r, divisor as i32 - 1 - r].into_iter().enumerate() {
                columns[i].push(v);
            }
        }
        let range_values = ranges(divisor)
            .iter()
            .map(|r| {
                columns[r.tensor]
                    .iter()
                    .map(|v| (i64::from(*v) + r.offset as i64) as u64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut polynomials = columns
            .iter()
            .enumerate()
            .map(|(i, v)| (tensor(i), MultilinearPolynomial::from(v.clone())))
            .collect::<BTreeMap<_, _>>();
        for r in ranges(divisor) {
            for d in 0..r.chunks() {
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(r.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        range_values[r.tensor]
                            .iter()
                            .map(|v| Some(u16::from(r.digit(*v, d))))
                            .collect(),
                        1 << r.chunk,
                    )),
                );
            }
        }
        Ok(Self {
            output: columns[if remainder { 2 } else { 1 }].clone(),
            polynomials,
            range_values,
        })
    }
}
pub(super) struct DivisionRegistration {
    pub tensors: [CommittedPoly; N],
    pub divisor: u32,
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
}
impl DivisionRegistration {
    pub(super) fn ranges(&self) -> Vec<Range> {
        ranges(self.divisor)
            .into_iter()
            .map(|mut r| {
                r.input = OpeningId::new(self.tensors[r.tensor], self.initial);
                r.namespace = self.namespace + r.tensor;
                r
            })
            .collect()
    }
    pub fn keys(&self) -> Vec<CommittedPoly> {
        self.tensors
            .into_iter()
            .chain(self.ranges().into_iter().flat_map(|r| {
                (0..r.chunks()).map(move |d| CommittedPoly::NodeOutputRaD(r.namespace, d))
            }))
            .collect()
    }
    fn params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> DivisionParams {
        let a: Fr = t.challenge_scalar();
        let b: Fr = t.challenge_scalar();
        let divisor = Fr::from(u64::from(self.divisor));
        DivisionParams {
            point: r.to_vec(),
            openings: self.tensors.map(|p| OpeningId::new(p, self.final_stage)),
            coefficients: [a, -a * divisor, b - a, b, -b * (divisor - Fr::from(1u64))],
        }
    }
    pub fn provers(
        &self,
        log_rows: usize,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>, ProofVerifyError> {
        if values.len() != N || values.iter().any(|v| v.len() != 1 << log_rows) {
            return Err(invalid("Invalid integer division range witness"));
        }
        let point: Vec<Fr> = t.challenge_vector(log_rows);
        let params = self.params(&point, t);
        let evaluations = self.tensors.map(|p| polynomials[&p].evaluate(&point));
        let mut result: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> =
            vec![Box::new(DivisionProver {
                params,
                evaluations,
            })];
        for r in self.ranges() {
            a.append_dense(t, r.input, point.clone(), evaluations[r.tensor]);
            let params = r.params(&point, t);
            result.push(Box::new(r.prover(&values[r.tensor], params)));
        }
        Ok(result)
    }
    pub fn verifiers(
        &self,
        log_rows: usize,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let point: Vec<Fr> = t.challenge_vector(log_rows);
        let params = self.params(&point, t);
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> =
            vec![Box::new(DivisionVerifier(params))];
        for r in self.ranges() {
            a.append_dense(t, r.input, point.clone());
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                r.params(&point, t),
            )));
        }
        result
    }
}
#[derive(Clone)]
struct DivisionParams {
    point: Vec<Fr>,
    openings: [OpeningId; N],
    coefficients: [Fr; N + 1],
}
impl SumcheckInstanceParams<Fr> for DivisionParams {
    fn degree(&self) -> usize {
        1
    }
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
            (0..N)
                .map(|i| {
                    ProductTerm::scaled(
                        ValueSource::Challenge(i),
                        vec![ValueSource::Opening(self.openings[i])],
                    )
                })
                .chain(std::iter::once(ProductTerm::single(
                    ValueSource::Challenge(N),
                )))
                .collect(),
        ))
    }
    fn output_constraint_challenge_values(&self, _: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        self.coefficients.to_vec()
    }
}
struct DivisionProver {
    params: DivisionParams,
    evaluations: [Fr; N],
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for DivisionProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        unreachable!("Division identities have no sumcheck rounds")
    }
    fn ingest_challenge(&mut self, _: <Fr as JoltField>::Challenge, _: usize) {
        unreachable!("Division identities have no sumcheck rounds")
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..N {
            a.append_dense(
                t,
                self.params.openings[i],
                self.params.point.clone(),
                self.evaluations[i],
            );
        }
    }
}
struct DivisionVerifier(DivisionParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for DivisionVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        _: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        (0..N)
            .map(|i| {
                self.0.coefficients[i] * a.get_committed_polynomial_opening(self.0.openings[i]).1
            })
            .sum::<Fr>()
            + self.0.coefficients[N]
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for id in self.0.openings {
            a.append_dense(t, id, self.0.point.clone());
        }
    }
}
