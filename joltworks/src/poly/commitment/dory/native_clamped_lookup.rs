//! Clamp a hidden signed input to a public table interval, then recover its index.
//!
//! The mandatory table relation separately proves index in [0, K). This module
//! proves x = lower + index - lo + hi, lo*index = 0 and hi*(K-1-index) = 0.
//! Input is signed32 and gaps unsigned32. Together these relations prove the
//! unique signed clamp and address translation without a full signed32 table.

use super::native_mul::Range;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
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
fn ranges() -> Vec<Range> {
    vec![
        Range::new(0, 32, 1 << 31),
        Range::new(2, 32, 0),
        Range::new(3, 32, 0),
    ]
}

pub(super) struct ClampedLookupWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub indices: Vec<usize>,
}
impl ClampedLookupWitness {
    pub fn new(input: &[i32], lower: i32, table_len: usize) -> Result<Self, ProofVerifyError> {
        if input.is_empty()
            || !input.len().is_power_of_two()
            || table_len < 2
            || !table_len.is_power_of_two()
            || table_len.ilog2() > 31
        {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Invalid clamped lookup shape or interval".into(),
            ));
        }
        let upper = i64::from(lower) + table_len as i64 - 1;
        if upper > i64::from(i32::MAX) {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Table interval exceeds signed32".into(),
            ));
        }
        let mut indices = Vec::with_capacity(input.len());
        let mut lo = Vec::with_capacity(input.len());
        let mut hi = Vec::with_capacity(input.len());
        for x in input {
            let x = i64::from(*x);
            let y = x.clamp(i64::from(lower), upper);
            indices.push((y - i64::from(lower)) as usize);
            lo.push((y - x).max(0) as u64);
            hi.push((x - y).max(0) as u64);
        }
        let index_values = indices.iter().map(|i| *i as u64).collect::<Vec<_>>();
        let range_values = vec![
            input
                .iter()
                .map(|x| (i64::from(*x) + (1i64 << 31)) as u64)
                .collect(),
            index_values.clone(),
            lo.clone(),
            hi.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(input.to_vec())),
            (tensor(1), MultilinearPolynomial::from(index_values)),
            (tensor(2), MultilinearPolynomial::from(lo)),
            (tensor(3), MultilinearPolynomial::from(hi)),
        ]);
        for r in ranges() {
            for d in 0..r.chunks() {
                let indices = range_values[r.tensor]
                    .iter()
                    .map(|v| Some(u16::from(r.digit(*v, d))))
                    .collect();
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(r.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        indices,
                        1 << r.chunk,
                    )),
                );
            }
        }
        Ok(Self {
            polynomials,
            range_values,
            indices,
        })
    }
}

/// All identifiers are reconstructed from the verifier's registered graph.
pub(super) struct ClampedLookupRegistration {
    pub tensors: [CommittedPoly; N],
    pub lower: i32,
    pub table_len: usize,
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
}
impl ClampedLookupRegistration {
    fn source(&self, i: usize) -> OpeningId {
        OpeningId::new(self.tensors[i], self.initial)
    }
    pub(super) fn ranges(&self) -> Vec<Range> {
        ranges()
            .into_iter()
            .map(|mut r| {
                r.input = self.source(r.tensor);
                r.namespace = self.namespace + r.tensor;
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
    fn params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> ClampIndexParams {
        let mut params = ClampIndexParams::new(r, self.lower, self.table_len, t);
        params.openings = self.tensors.map(|id| OpeningId::new(id, self.final_stage));
        params
    }
    pub fn provers(
        &self,
        log_rows: usize,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> {
        let r: Vec<Fr> = t.challenge_vector(log_rows);
        for i in 0..N {
            a.append_dense(
                t,
                self.source(i),
                r.clone(),
                polynomials[&self.tensors[i]].evaluate(&r),
            );
        }
        let arithmetic = ClampIndexProver {
            params: self.params(&r, t),
            values: self.tensors.map(|id| polynomials[&id].clone()),
            eq: MultilinearPolynomial::from(EqPolynomial::<Fr>::evals(&r)),
        };
        let mut result: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> =
            vec![Box::new(arithmetic)];
        for range in self.ranges() {
            let params = range.params(&r, t);
            result.push(Box::new(range.prover(&values[range.tensor], params)));
        }
        result
    }
    pub fn verifiers(
        &self,
        log_rows: usize,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let r: Vec<Fr> = t.challenge_vector(log_rows);
        for i in 0..N {
            a.append_dense(t, self.source(i), r.clone());
        }
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> =
            vec![Box::new(ClampIndexVerifier(self.params(&r, t)))];
        for range in self.ranges() {
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&r, t),
            )));
        }
        result
    }
}

#[derive(Clone)]
struct ClampIndexParams {
    r: Vec<Fr>,
    lower: i32,
    table_len: usize,
    gamma: [Fr; 3],
    openings: [OpeningId; N],
}
impl ClampIndexParams {
    fn new(r: &[Fr], lower: i32, table_len: usize, t: &mut Blake2bTranscript) -> Self {
        Self {
            r: r.to_vec(),
            openings: std::array::from_fn(|i| {
                OpeningId::new(tensor(i), SumcheckId::NodeExecution(1))
            }),
            lower,
            table_len,
            gamma: [
                t.challenge_scalar(),
                t.challenge_scalar(),
                t.challenge_scalar(),
            ],
        }
    }
    fn evaluate(&self, v: &[Fr; N]) -> Fr {
        let [x, index, lo, hi] = *v;
        self.gamma[0] * (x - Fr::from_i64(i64::from(self.lower)) - index + lo - hi)
            + self.gamma[1] * lo * index
            + self.gamma[2] * hi * (Fr::from((self.table_len - 1) as u64) - index)
    }
    fn coefficients(&self) -> Vec<Fr> {
        vec![
            self.gamma[0],
            -self.gamma[0],
            self.gamma[0],
            -self.gamma[0],
            self.gamma[1],
            -self.gamma[2],
            self.gamma[2] * Fr::from((self.table_len - 1) as u64),
            -self.gamma[0] * Fr::from_i64(i64::from(self.lower)),
        ]
    }
}

impl SumcheckInstanceParams<Fr> for ClampIndexParams {
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.r.len()
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
        let value = |i| ValueSource::Opening(self.openings[i]);
        let factors = [
            vec![value(0)],
            vec![value(1)],
            vec![value(2)],
            vec![value(3)],
            vec![value(2), value(1)],
            vec![value(3), value(1)],
            vec![value(3)],
            vec![ValueSource::Constant(1)],
        ];
        Some(OutputClaimConstraint::sum_of_products(
            factors
                .into_iter()
                .enumerate()
                .map(|(i, values)| ProductTerm::scaled(ValueSource::Challenge(i), values))
                .collect(),
        ))
    }

    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        let eq = EqPolynomial::mle(r, &self.r);
        self.coefficients().into_iter().map(|c| eq * c).collect()
    }
}
#[derive(allocative::Allocative)]
struct ClampIndexProver {
    #[allocative(skip)]
    params: ClampIndexParams,
    values: [MultilinearPolynomial<Fr>; N],
    eq: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ClampIndexProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        let half = self.eq.len() / 2;
        let evals = (0..4)
            .map(|k| {
                let z = Fr::from(k as u64);
                (0..half)
                    .map(|i| {
                        let at = |p: &MultilinearPolynomial<Fr>| {
                            let a = p.get_bound_coeff(i);
                            a + z * (p.get_bound_coeff(i + half) - a)
                        };
                        at(&self.eq)
                            * self
                                .params
                                .evaluate(&std::array::from_fn(|j| at(&self.values[j])))
                    })
                    .sum()
            })
            .collect::<Vec<_>>();
        UniPoly::from_evals(&evals)
    }
    fn ingest_challenge(&mut self, r: <Fr as JoltField>::Challenge, _: usize) {
        self.eq.bind_parallel(r, BindingOrder::HighToLow);
        for p in &mut self.values {
            p.bind_parallel(r, BindingOrder::HighToLow);
        }
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..N {
            a.append_dense(
                t,
                self.params.openings[i],
                r.to_vec(),
                self.values[i].final_claim(),
            );
        }
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, f: &mut allocative::FlameGraphBuilder) {
        f.visit_root(self);
    }
}
struct ClampIndexVerifier(ClampIndexParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ClampIndexVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        EqPolynomial::mle(r, &self.0.r)
            * self.0.evaluate(&std::array::from_fn(|j| {
                a.get_committed_polynomial_opening(self.0.openings[j]).1
            }))
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..N {
            a.append_dense(t, self.0.openings[i], r.to_vec());
        }
    }
}
