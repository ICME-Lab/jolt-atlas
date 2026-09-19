//! Prove the maximum along the last axis without revealing its position.
//! Signed input/output ranges and a nonnegative gap prove m >= x. A Boolean
//! selector has sum one per row and can select only zero gaps, proving m is
//! attained. The selector does not prove a tie-breaking rule or token selection.

use super::{native_mul::Range, native_reduce::shape_bits};
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
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
pub(super) fn ranges(bits: u8) -> Vec<Range> {
    vec![
        Range::new(0, usize::from(bits), 1u64 << (bits - 1)),
        Range::new(1, usize::from(bits), 1u64 << (bits - 1)),
        Range::new(2, 1, 0),
        Range::new(3, 32, 0),
    ]
}
#[derive(Clone)]
pub(super) struct Maximum {
    pub log_input: usize,
    pub log_output: usize,
    pub output_shape: Vec<usize>,
    pub bits: u8,
}
impl Maximum {
    pub fn new(shape: &[usize], bits: u8) -> Result<Self, ProofVerifyError> {
        let log_input = shape_bits(shape)?;
        if shape.is_empty() || !(2..=32).contains(&bits) {
            return Err(invalid("Invalid maximum rank or integer width"));
        }
        let mut output_shape = shape.to_vec();
        *output_shape.last_mut().unwrap() = 1;
        let log_output = shape_bits(&output_shape)?;
        Ok(Self {
            log_input,
            log_output,
            output_shape,
            bits,
        })
    }
    fn row_size(&self) -> usize {
        1 << (self.log_input - self.log_output)
    }
    fn point<'a, T>(&self, j: usize, r: &'a [T]) -> &'a [T] {
        if j == 1 {
            &r[..self.log_output]
        } else {
            r
        }
    }
}
pub(super) struct MaxWitness {
    pub output: Vec<i32>,
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
}
impl MaxWitness {
    pub fn new(input: &[i32], layout: &Maximum) -> Result<Self, ProofVerifyError> {
        let bound = 1i64 << (layout.bits - 1);
        if input.len() != 1 << layout.log_input
            || input
                .iter()
                .any(|x| i64::from(*x) < -bound || i64::from(*x) >= bound)
        {
            return Err(invalid(
                "Maximum input outside registered shape or signed range",
            ));
        }
        let mut output = Vec::with_capacity(1 << layout.log_output);
        let mut selected = Vec::with_capacity(input.len());
        let mut gaps = Vec::with_capacity(input.len());
        for row in input.chunks_exact(layout.row_size()) {
            let max = *row.iter().max().unwrap();
            output.push(max);
            let mut found = false;
            for x in row {
                let p = *x == max && !found;
                found |= p;
                selected.push(u64::from(p));
                gaps.push((i64::from(max) - i64::from(*x)) as u64);
            }
        }
        let range_values = vec![
            input
                .iter()
                .map(|x| (i64::from(*x) + bound) as u64)
                .collect(),
            output
                .iter()
                .map(|x| (i64::from(*x) + bound) as u64)
                .collect(),
            selected.clone(),
            gaps.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(input.to_vec())),
            (tensor(1), MultilinearPolynomial::from(output.clone())),
            (tensor(2), MultilinearPolynomial::from(selected)),
            (tensor(3), MultilinearPolynomial::from(gaps)),
        ]);
        for range in ranges(layout.bits) {
            for d in 0..range.chunks() {
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(range.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        range_values[range.tensor]
                            .iter()
                            .map(|v| Some(u16::from(range.digit(*v, d))))
                            .collect(),
                        1 << range.chunk,
                    )),
                );
            }
        }
        Ok(Self {
            output,
            polynomials,
            range_values,
        })
    }
}
pub(super) struct MaxRegistration {
    pub tensors: [CommittedPoly; N],
    pub initial: SumcheckId,
    pub final_stages: [SumcheckId; 2],
    pub namespace: usize,
    pub layout: Maximum,
}
impl MaxRegistration {
    fn source(&self, j: usize) -> OpeningId {
        OpeningId::new(self.tensors[j], self.initial)
    }
    pub(super) fn ranges(&self) -> Vec<Range> {
        ranges(self.layout.bits)
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
    fn params(&self, r: &[Fr], selected: bool, t: &mut Blake2bTranscript) -> MaxParams {
        MaxParams {
            r: r.to_vec(),
            layout: self.layout.clone(),
            selected,
            gamma: std::array::from_fn(|_| t.challenge_scalar()),
            openings: self
                .tensors
                .map(|id| OpeningId::new(id, self.final_stages[usize::from(selected)])),
        }
    }
    pub fn provers(
        &self,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>, ProofVerifyError> {
        if values.len() != N
            || values.iter().enumerate().any(|(j, v)| {
                v.len()
                    != 1 << if j == 1 {
                        self.layout.log_output
                    } else {
                        self.layout.log_input
                    }
            })
        {
            return Err(invalid("Invalid maximum range witness"));
        }
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        for j in 0..N {
            let point = self.layout.point(j, &r);
            a.append_dense(
                t,
                self.source(j),
                point.to_vec(),
                polynomials[&self.tensors[j]].evaluate(point),
            );
        }
        let pointwise = self.params(&r, false, t);
        let selected = self.params(&r[..self.layout.log_output], true, t);
        let expanded = MultilinearPolynomial::from(
            (0..1 << self.layout.log_input)
                .map(|i| polynomials[&self.tensors[1]].get_bound_coeff(i / self.layout.row_size()))
                .collect::<Vec<_>>(),
        );
        let polys: [MultilinearPolynomial<Fr>; N] = std::array::from_fn(|j| {
            if j == 1 {
                expanded.clone()
            } else {
                polynomials[&self.tensors[j]].clone()
            }
        });
        let row_eq = EqPolynomial::<Fr>::evals(&r[..self.layout.log_output]);
        let selector = MultilinearPolynomial::from(
            (0..1 << self.layout.log_input)
                .map(|i| row_eq[i / self.layout.row_size()])
                .collect::<Vec<_>>(),
        );
        let mut result: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> = vec![
            Box::new(MaxProver {
                params: pointwise,
                values: polys.clone(),
                eq: MultilinearPolynomial::from(EqPolynomial::<Fr>::evals(&r)),
            }),
            Box::new(MaxProver {
                params: selected,
                values: polys,
                eq: selector,
            }),
        ];
        for range in self.ranges() {
            let params = range.params(self.layout.point(range.tensor, &r), t);
            result.push(Box::new(range.prover(&values[range.tensor], params)));
        }
        Ok(result)
    }
    pub fn verifiers(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        for j in 0..N {
            a.append_dense(t, self.source(j), self.layout.point(j, &r).to_vec());
        }
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> = vec![
            Box::new(MaxVerifier(self.params(&r, false, t))),
            Box::new(MaxVerifier(self.params(
                &r[..self.layout.log_output],
                true,
                t,
            ))),
        ];
        for range in self.ranges() {
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(self.layout.point(range.tensor, &r), t),
            )));
        }
        result
    }
}
#[derive(Clone)]
struct MaxParams {
    r: Vec<Fr>,
    layout: Maximum,
    selected: bool,
    gamma: [Fr; 2],
    openings: [OpeningId; N],
}
impl MaxParams {
    fn evaluate(&self, v: &[Fr; N]) -> Fr {
        if self.selected {
            v[2]
        } else {
            self.gamma[0] * (v[0] + v[3] - v[1]) + self.gamma[1] * v[2] * v[3]
        }
    }
    fn eq(&self, r: &[<Fr as JoltField>::Challenge]) -> Fr {
        EqPolynomial::mle(
            if self.selected {
                &r[..self.layout.log_output]
            } else {
                r
            },
            &self.r,
        )
    }
    fn active(&self) -> Vec<usize> {
        if self.selected {
            vec![2]
        } else {
            vec![0, 1, 2, 3]
        }
    }
}
impl SumcheckInstanceParams<Fr> for MaxParams {
    fn degree(&self) -> usize {
        if self.selected {
            2
        } else {
            3
        }
    }
    fn num_rounds(&self) -> usize {
        self.layout.log_input
    }
    fn input_claim(&self, _: &dyn OpeningAccumulator<Fr>) -> Fr {
        Fr::from(u64::from(self.selected))
    }
    fn normalize_opening_point(&self, r: &[Fr]) -> OpeningPoint<BIG_ENDIAN, Fr> {
        r.to_vec().into()
    }
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::sum_of_products(vec![ProductTerm::single(ValueSource::Constant(
            i128::from(self.selected),
        ))])
    }
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<Fr>) -> Vec<Fr> {
        vec![]
    }
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let value = |j| ValueSource::Opening(self.openings[j]);
        let factors = if self.selected {
            vec![vec![value(2)]]
        } else {
            vec![
                vec![value(0)],
                vec![value(3)],
                vec![value(1)],
                vec![value(2), value(3)],
            ]
        };
        Some(OutputClaimConstraint::sum_of_products(
            factors
                .into_iter()
                .enumerate()
                .map(|(j, v)| ProductTerm::scaled(ValueSource::Challenge(j), v))
                .collect(),
        ))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        let eq = self.eq(r);
        if self.selected {
            vec![eq]
        } else {
            vec![
                eq * self.gamma[0],
                eq * self.gamma[0],
                -eq * self.gamma[0],
                eq * self.gamma[1],
            ]
        }
    }
}
#[derive(allocative::Allocative)]
struct MaxProver {
    #[allocative(skip)]
    params: MaxParams,
    values: [MultilinearPolynomial<Fr>; N],
    eq: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for MaxProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        let half = self.eq.len() / 2;
        let evals = (0..=self.params.degree())
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
        for i in self.params.active() {
            a.append_dense(
                t,
                self.params.openings[i],
                self.params.layout.point(i, r).to_vec(),
                self.values[i].final_claim(),
            );
        }
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, f: &mut allocative::FlameGraphBuilder) {
        f.visit_root(self);
    }
}
struct MaxVerifier(MaxParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for MaxVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        let mut values = [Fr::zero(); N];
        for j in self.0.active() {
            values[j] = a.get_committed_polynomial_opening(self.0.openings[j]).1;
        }
        self.0.eq(r) * self.0.evaluate(&values)
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in self.0.active() {
            a.append_dense(t, self.0.openings[i], self.0.layout.point(i, r).to_vec());
        }
    }
}
