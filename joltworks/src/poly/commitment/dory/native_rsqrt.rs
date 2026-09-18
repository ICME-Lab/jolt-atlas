//! Exact integer reciprocal square root with hidden ranges and sign selection.
//!
//! For A = 2^(3*scale), positive x requires x*y^2 <= A < x*(y+1)^2.
//! Nonpositive x requires y = 0. All identities use the actual committed input.
//! Scales 0..=20 match the Atlas integer kernel without signed overflow.

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

const N: usize = 6;
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
pub(super) fn ranges() -> Vec<Range> {
    vec![
        Range::new(0, 32, 1 << 31),
        Range::new(1, 32, 0),
        Range::new(2, 1, 0),
        Range::new(3, 32, 0),
        Range::new(4, 64, 0),
        Range::new(5, 64, 0),
    ]
}

pub(super) struct RsqrtWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub output: Vec<i32>,
}
impl RsqrtWitness {
    pub fn new(input: &[i32], scale: u8) -> Result<Self, ProofVerifyError> {
        if input.is_empty() || !input.len().is_power_of_two() || scale > 20 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Invalid reciprocal square root shape or scale".into(),
            ));
        }
        let a = 1u64 << (3 * scale);
        let mut columns: [Vec<u64>; N] = std::array::from_fn(|_| Vec::with_capacity(input.len()));
        let mut output = Vec::with_capacity(input.len());
        for x in input {
            let x = i64::from(*x);
            let row = if x <= 0 {
                [0, 0, 0, (-x) as u64, 0, 0]
            } else {
                let x = x as u64;
                let y = (a / x).isqrt();
                // Wide intermediates keep witness construction independent of
                // field reduction. The valid integer gaps both fit u64.
                let lower = u128::from(a) - u128::from(x) * u128::from(y).pow(2);
                let upper = u128::from(x) * u128::from(y + 1).pow(2) - u128::from(a) - 1;
                [
                    0,
                    y,
                    1,
                    x - 1,
                    u64::try_from(lower).unwrap(),
                    u64::try_from(upper).unwrap(),
                ]
            };
            for (j, column) in columns.iter_mut().enumerate() {
                column.push(if j == 0 {
                    (x + (1i64 << 31)) as u64
                } else {
                    row[j]
                });
            }
            output.push(i32::try_from(row[1]).unwrap());
        }
        let mut polynomials = BTreeMap::new();
        polynomials.insert(tensor(0), MultilinearPolynomial::from(input.to_vec()));
        for (j, column) in columns.iter().enumerate().skip(1) {
            polynomials.insert(tensor(j), MultilinearPolynomial::from(column.clone()));
        }
        for r in ranges() {
            for d in 0..r.chunks() {
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(r.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        columns[r.tensor]
                            .iter()
                            .map(|v| Some(u16::from(r.digit(*v, d))))
                            .collect(),
                        1 << r.chunk,
                    )),
                );
            }
        }
        Ok(Self {
            polynomials,
            range_values: columns.into(),
            output,
        })
    }
}

/// All identifiers are reconstructed from the verifier's registered graph.
pub(super) struct RsqrtRegistration {
    pub tensors: [CommittedPoly; N],
    pub scale: u8,
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
}
impl RsqrtRegistration {
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
    fn params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> RsqrtParams {
        let mut params = RsqrtParams::new(r, self.scale, t);
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
        let arithmetic = RsqrtProver {
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
            vec![Box::new(RsqrtVerifier(self.params(&r, t)))];
        for range in self.ranges() {
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&r, t),
            )));
        }
        result
    }
}

#[derive(Clone)]
struct RsqrtParams {
    r: Vec<Fr>,
    a: Fr,
    gamma: [Fr; 4],
    openings: [OpeningId; N],
}
impl RsqrtParams {
    fn new(r: &[Fr], scale: u8, t: &mut Blake2bTranscript) -> Self {
        Self {
            r: r.to_vec(),
            openings: std::array::from_fn(|i| {
                OpeningId::new(tensor(i), SumcheckId::NodeExecution(1))
            }),
            a: Fr::from(1u64 << (3 * scale)),
            gamma: std::array::from_fn(|_| t.challenge_scalar()),
        }
    }
    fn evaluate(&self, v: &[Fr; N]) -> Fr {
        let [x, y, b, sign_gap, lo, hi] = *v;
        let one = Fr::from(1u64);
        self.gamma[0] * (x - (b + b - one) * sign_gap - b)
            + self.gamma[1] * b * (self.a - x * y * y - lo)
            + self.gamma[2] * b * (x * (y + one) * (y + one) - self.a - one - hi)
            + self.gamma[3] * (one - b) * y
    }
    fn coefficients(&self) -> Vec<Fr> {
        let [g0, g1, g2, g3] = self.gamma;
        vec![
            g0,
            g0,
            g1 * self.a - g2 * (self.a + Fr::from(1u64)) - g0,
            -g0 - g0,
            g2 - g1,
            -g1,
            g2 + g2,
            g2,
            -g2,
            g3,
            -g3,
        ]
    }
}

impl SumcheckInstanceParams<Fr> for RsqrtParams {
    fn degree(&self) -> usize {
        5
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
            vec![value(3)],
            vec![value(2)],
            vec![value(2), value(3)],
            vec![value(2), value(0), value(1), value(1)],
            vec![value(2), value(4)],
            vec![value(2), value(0), value(1)],
            vec![value(2), value(0)],
            vec![value(2), value(5)],
            vec![value(1)],
            vec![value(2), value(1)],
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
struct RsqrtProver {
    #[allocative(skip)]
    params: RsqrtParams,
    values: [MultilinearPolynomial<Fr>; N],
    eq: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for RsqrtProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        let half = self.eq.len() / 2;
        let evals = (0..6)
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
struct RsqrtVerifier(RsqrtParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for RsqrtVerifier {
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
