//! Exact Atlas addition and subtraction with a hidden signed integer clamp.
//!
//! The relation is a +/- b = y - low + high, low*(y-MIN) = 0 and
//! high*(MAX-y) = 0. Both operands and y are signed 32 bit integers. Gaps
//! are unsigned 32 bit integers. These bounds prevent field wraparound and
//! make the identities equivalent to the integer clamp. The graph shares
//! its actual input and output commitments with every neighboring operator.

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

const N: usize = 5;
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
fn ranges() -> Vec<Range> {
    (0..N)
        .map(|i| Range::new(i, 32, if i < 3 { 1 << 31 } else { 0 }))
        .collect()
}

pub(super) struct AddWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub output: Vec<i32>,
}
impl AddWitness {
    pub fn new(left: &[i32], right: &[i32], subtract: bool) -> Result<Self, ProofVerifyError> {
        if left.is_empty() || !left.len().is_power_of_two() || left.len() != right.len() {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Invalid addition witness shape".into(),
            ));
        }
        let mut output = Vec::with_capacity(left.len());
        let mut lo = Vec::with_capacity(left.len());
        let mut hi = Vec::with_capacity(left.len());
        for (a, b) in left.iter().zip(right) {
            let wide = if subtract {
                i64::from(*a) - i64::from(*b)
            } else {
                i64::from(*a) + i64::from(*b)
            };
            let y = wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
            output.push(y as i32);
            lo.push((y - wide).max(0) as u64);
            hi.push((wide - y).max(0) as u64);
        }
        let signed = |v: &[i32]| {
            v.iter()
                .map(|x| (i64::from(*x) + (1i64 << 31)) as u64)
                .collect::<Vec<_>>()
        };
        let range_values = vec![
            signed(left),
            signed(right),
            signed(&output),
            lo.clone(),
            hi.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(left.to_vec())),
            (tensor(1), MultilinearPolynomial::from(right.to_vec())),
            (tensor(2), MultilinearPolynomial::from(output.clone())),
            (tensor(3), MultilinearPolynomial::from(lo)),
            (tensor(4), MultilinearPolynomial::from(hi)),
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
            output,
        })
    }
}

/// All identifiers are reconstructed from the verifier's registered graph.
pub(super) struct AddRegistration {
    pub tensors: [CommittedPoly; N],
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
}
impl AddRegistration {
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
    fn params(&self, r: &[Fr], subtract: bool, t: &mut Blake2bTranscript) -> ArithmeticParams {
        let mut params = ArithmeticParams::new(r, subtract, t);
        params.openings = self.tensors.map(|id| OpeningId::new(id, self.final_stage));
        params
    }
    pub fn provers(
        &self,
        subtract: bool,
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
        let arithmetic = ArithmeticProver {
            params: self.params(&r, subtract, t),
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
        subtract: bool,
        log_rows: usize,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let r: Vec<Fr> = t.challenge_vector(log_rows);
        for i in 0..N {
            a.append_dense(t, self.source(i), r.clone());
        }
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> =
            vec![Box::new(ArithmeticVerifier(self.params(&r, subtract, t)))];
        for range in self.ranges() {
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&r, t),
            )));
        }
        result
    }
}

#[derive(Clone)]
struct ArithmeticParams {
    r: Vec<Fr>,
    subtract: bool,
    gamma: [Fr; 3],
    openings: [OpeningId; N],
}
impl ArithmeticParams {
    fn new(r: &[Fr], subtract: bool, t: &mut Blake2bTranscript) -> Self {
        Self {
            r: r.to_vec(),
            openings: std::array::from_fn(|i| {
                OpeningId::new(tensor(i), SumcheckId::NodeExecution(1))
            }),
            subtract,
            gamma: [
                t.challenge_scalar(),
                t.challenge_scalar(),
                t.challenge_scalar(),
            ],
        }
    }
    fn evaluate(&self, v: &[Fr; N]) -> Fr {
        let [a, b, y, lo, hi] = *v;
        let sum = if self.subtract { a - b } else { a + b };
        self.gamma[0] * (sum - y + lo - hi)
            + self.gamma[1] * lo * (y + Fr::from(1u64 << 31))
            + self.gamma[2] * hi * (Fr::from(i32::MAX as u64) - y)
    }
    fn coefficients(&self) -> Vec<Fr> {
        vec![
            self.gamma[0],
            if self.subtract {
                -self.gamma[0]
            } else {
                self.gamma[0]
            },
            -self.gamma[0],
            self.gamma[0],
            -self.gamma[0],
            self.gamma[1],
            self.gamma[1] * Fr::from(1u64 << 31),
            -self.gamma[2],
            self.gamma[2] * Fr::from(i32::MAX as u64),
        ]
    }
}

impl SumcheckInstanceParams<Fr> for ArithmeticParams {
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
        let factors = [
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![3, 2],
            vec![3],
            vec![4, 2],
            vec![4],
        ];
        Some(OutputClaimConstraint::sum_of_products(
            factors
                .into_iter()
                .enumerate()
                .map(|(i, indices)| {
                    ProductTerm::scaled(
                        ValueSource::Challenge(i),
                        indices
                            .into_iter()
                            .map(|j| ValueSource::Opening(self.openings[j]))
                            .collect(),
                    )
                })
                .collect(),
        ))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        let eq = EqPolynomial::mle(r, &self.r);
        self.coefficients().into_iter().map(|c| eq * c).collect()
    }
}
#[derive(allocative::Allocative)]
struct ArithmeticProver {
    #[allocative(skip)]
    params: ArithmeticParams,
    values: [MultilinearPolynomial<Fr>; N],
    eq: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ArithmeticProver {
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
        crate::subprotocols::opening_reduction::reclaim_opening_coefficients(&mut self.eq);
        for p in &mut self.values {
            p.bind_parallel(r, BindingOrder::HighToLow);
            crate::subprotocols::opening_reduction::reclaim_opening_coefficients(p);
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
struct ArithmeticVerifier(ArithmeticParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ArithmeticVerifier {
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
