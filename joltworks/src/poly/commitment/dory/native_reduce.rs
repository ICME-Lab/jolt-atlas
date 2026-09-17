//! Hidden exact reductions over registered tensor axes, followed by one clamp.
//! Mean of squares also proves division by the shape-derived count and scale.
//! Overflow outside the signed 64 bit Atlas accumulator domain is rejected.

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

type Provers = Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>;
type Verifiers = Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>>;

fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}

/// Exact power of two dimensions, including a scalar with no dimensions.
/// The size bound also bounds the integer reduction before field conversion.
pub(super) fn shape_bits(shape: &[usize]) -> Result<usize, ProofVerifyError> {
    if shape.len() > 32 || shape.iter().any(|d| !d.is_power_of_two()) {
        return Err(invalid("Tensor dimensions must be positive powers of two"));
    }
    let bits: usize = shape.iter().map(|d| d.ilog2() as usize).sum();
    if bits > 30 {
        return Err(invalid("Native tensor exceeds the registered size bound"));
    }
    Ok(bits)
}

#[derive(Clone)]
pub(super) struct Reduction {
    pub output_shape: Vec<usize>,
    pub kept: Vec<usize>,
    pub log_input: usize,
    pub log_output: usize,
    pub square: bool,
    pub shift: u8,
}
impl Reduction {
    pub fn new(
        shape: &[usize],
        axes: &[usize],
        mean_scale: Option<u8>,
    ) -> Result<Self, ProofVerifyError> {
        let log_input = shape_bits(shape)?;
        if axes.iter().any(|a| *a >= shape.len()) || axes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(
                "Reduction axes must be distinct, ordered and in range",
            ));
        }
        let mut kept = vec![];
        let mut output_shape = shape.to_vec();
        let mut first = 0;
        for (axis, size) in shape.iter().enumerate() {
            let bits = size.ilog2() as usize;
            if axes.contains(&axis) {
                output_shape[axis] = 1;
            } else {
                kept.extend(first..first + bits);
            }
            first += bits;
        }
        let log_output = kept.len();
        let shift = if let Some(scale) = mean_scale {
            let total = log_input - log_output + usize::from(scale);
            if total > 30 {
                return Err(invalid(
                    "Mean divisor exceeds the supported Atlas remainder range",
                ));
            }
            total as u8
        } else {
            0
        };
        Ok(Self {
            output_shape,
            kept,
            log_input,
            log_output,
            square: mean_scale.is_some(),
            shift,
        })
    }
    pub fn columns(&self) -> usize {
        if self.shift == 0 {
            5
        } else {
            6
        }
    }
    pub fn output_index(&self, index: usize) -> usize {
        self.kept.iter().fold(0, |out, bit| {
            (out << 1) | ((index >> (self.log_input - 1 - bit)) & 1)
        })
    }
    fn ranges(&self) -> Vec<Range> {
        let mut ranges = vec![
            Range::new(0, 32, 1 << 31),
            Range::new(1, 32, 1 << 31),
            Range::new(2, 64, 1 << 63),
            Range::new(3, 64, 0),
            Range::new(4, 64, 0),
        ];
        if self.shift > 0 {
            ranges.push(Range::new(5, usize::from(self.shift), 0));
        }
        ranges
    }
}

pub(super) struct ReductionWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub output: Vec<i32>,
}
impl ReductionWitness {
    pub fn new(input: &[i32], reduction: &Reduction) -> Result<Self, ProofVerifyError> {
        if input.len() != 1 << reduction.log_input {
            return Err(invalid("Reduction input shape mismatch"));
        }
        let mut sums = vec![0i128; 1 << reduction.log_output];
        for (i, value) in input.iter().enumerate() {
            let v = i128::from(*value);
            sums[reduction.output_index(i)] += if reduction.square { v * v } else { v };
        }
        let acc = sums
            .into_iter()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("Exact Atlas reduction accumulator exceeds signed 64 bits"))?;
        let divisor = 1i128 << reduction.shift;
        let mut output = vec![];
        let mut lo = vec![];
        let mut hi = vec![];
        let mut rem = vec![];
        for value in &acc {
            let q = i128::from(*value).div_euclid(divisor);
            let y = q.clamp(i128::from(i32::MIN), i128::from(i32::MAX));
            output.push(y as i32);
            lo.push((y - q).max(0) as u64);
            hi.push((q - y).max(0) as u64);
            rem.push(i128::from(*value).rem_euclid(divisor) as u64);
        }
        let signed = |v: &[i32]| {
            v.iter()
                .map(|x| (i64::from(*x) + (1i64 << 31)) as u64)
                .collect::<Vec<_>>()
        };
        let mut range_values = vec![
            signed(input),
            signed(&output),
            acc.iter()
                .map(|v| (i128::from(*v) + (1i128 << 63)) as u64)
                .collect(),
            lo.clone(),
            hi.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(input.to_vec())),
            (tensor(1), MultilinearPolynomial::from(output.clone())),
            (
                tensor(2),
                MultilinearPolynomial::from(
                    acc.iter().map(|v| Fr::from_i64(*v)).collect::<Vec<_>>(),
                ),
            ),
            (tensor(3), MultilinearPolynomial::from(lo)),
            (tensor(4), MultilinearPolynomial::from(hi)),
        ]);
        if reduction.shift > 0 {
            range_values.push(rem.clone());
            polynomials.insert(tensor(5), MultilinearPolynomial::from(rem));
        }
        for r in reduction.ranges() {
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

pub(super) struct ReductionRegistration {
    /// Input, output, accumulator, lower gap, upper gap, optional remainder.
    pub tensors: [CommittedPoly; 6],
    pub initial: SumcheckId,
    pub reduction_stage: SumcheckId,
    pub clamp_stage: SumcheckId,
    pub namespace: usize,
    pub reduction: Reduction,
}
impl ReductionRegistration {
    fn source(&self, i: usize) -> OpeningId {
        OpeningId::new(self.tensors[i], self.initial)
    }
    fn ranges(&self) -> Vec<Range> {
        self.reduction
            .ranges()
            .into_iter()
            .map(|mut r| {
                r.input = self.source(r.tensor);
                r.namespace = self.namespace + r.tensor;
                r
            })
            .collect()
    }
    pub fn keys(&self) -> Vec<CommittedPoly> {
        self.tensors[..self.reduction.columns()]
            .iter()
            .copied()
            .chain(self.ranges().into_iter().flat_map(|r| {
                (0..r.chunks()).map(move |d| CommittedPoly::NodeOutputRaD(r.namespace, d))
            }))
            .collect()
    }
    fn reduction_params(&self, r: &[Fr]) -> ReductionParams {
        ReductionParams {
            log_input: self.reduction.log_input,
            kept: self.reduction.kept.clone(),
            r_output: r.to_vec(),
            square: self.reduction.square,
            accumulator: self.source(2),
            input: OpeningId::new(self.tensors[0], self.reduction_stage),
        }
    }
    fn clamp_params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> ClampParams {
        ClampParams {
            r: r.to_vec(),
            divisor: Fr::from(1u64 << self.reduction.shift),
            gamma: [
                t.challenge_scalar(),
                t.challenge_scalar(),
                t.challenge_scalar(),
            ],
            openings: self.tensors[1..self.reduction.columns()]
                .iter()
                .map(|id| OpeningId::new(*id, self.clamp_stage))
                .collect(),
        }
    }
    pub fn provers(
        &self,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Provers, ProofVerifyError> {
        if values.len() != self.reduction.columns()
            || values.iter().enumerate().any(|(i, v)| {
                v.len()
                    != 1 << if i == 0 {
                        self.reduction.log_input
                    } else {
                        self.reduction.log_output
                    }
            })
        {
            return Err(invalid("Reduction range witness shape mismatch"));
        }
        let r_input: Vec<Fr> = t.challenge_vector(self.reduction.log_input);
        let r_output: Vec<Fr> = t.challenge_vector(self.reduction.log_output);
        for i in 0..self.reduction.columns() {
            let r = if i == 0 { &r_input } else { &r_output };
            a.append_dense(
                t,
                self.source(i),
                r.clone(),
                polynomials[&self.tensors[i]].evaluate(r),
            );
        }
        let eq = EqPolynomial::<Fr>::evals(&r_output);
        let reduction = ReductionProver {
            params: self.reduction_params(&r_output),
            input: polynomials[&self.tensors[0]].clone(),
            selector: MultilinearPolynomial::from(
                (0..1 << self.reduction.log_input)
                    .map(|i| eq[self.reduction.output_index(i)])
                    .collect::<Vec<_>>(),
            ),
        };
        let clamp = ClampProver {
            params: self.clamp_params(&r_output, t),
            values: self.tensors[1..self.reduction.columns()]
                .iter()
                .map(|id| polynomials[id].clone())
                .collect(),
            eq: MultilinearPolynomial::from(eq),
        };
        let mut result: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> =
            vec![Box::new(reduction), Box::new(clamp)];
        for range in self.ranges() {
            let r = if range.tensor == 0 {
                &r_input
            } else {
                &r_output
            };
            let params = range.params(r, t);
            result.push(Box::new(range.prover(&values[range.tensor], params)));
        }
        Ok(result)
    }
    pub fn verifiers(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Verifiers {
        let r_input: Vec<Fr> = t.challenge_vector(self.reduction.log_input);
        let r_output: Vec<Fr> = t.challenge_vector(self.reduction.log_output);
        for i in 0..self.reduction.columns() {
            let r = if i == 0 { &r_input } else { &r_output };
            a.append_dense(t, self.source(i), r.clone());
        }
        let mut result: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> = vec![
            Box::new(ReductionVerifier(self.reduction_params(&r_output))),
            Box::new(ClampVerifier(self.clamp_params(&r_output, t))),
        ];
        for range in self.ranges() {
            let r = if range.tensor == 0 {
                &r_input
            } else {
                &r_output
            };
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(r, t),
            )));
        }
        result
    }
}

#[derive(Clone)]
struct ReductionParams {
    log_input: usize,
    kept: Vec<usize>,
    r_output: Vec<Fr>,
    square: bool,
    accumulator: OpeningId,
    input: OpeningId,
}
impl ReductionParams {
    fn selector(&self, r: &[<Fr as JoltField>::Challenge]) -> Fr {
        EqPolynomial::mle(
            &self.r_output,
            &self.kept.iter().map(|i| r[*i]).collect::<Vec<_>>(),
        )
    }
    fn value(&self, x: Fr) -> Fr {
        if self.square {
            x * x
        } else {
            x
        }
    }
}
impl SumcheckInstanceParams<Fr> for ReductionParams {
    fn degree(&self) -> usize {
        if self.square {
            3
        } else {
            2
        }
    }
    fn num_rounds(&self) -> usize {
        self.log_input
    }
    fn input_claim(&self, a: &dyn OpeningAccumulator<Fr>) -> Fr {
        a.get_committed_polynomial_opening(self.accumulator).1
    }
    fn normalize_opening_point(&self, r: &[Fr]) -> OpeningPoint<BIG_ENDIAN, Fr> {
        r.to_vec().into()
    }
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::sum_of_products(vec![ProductTerm::single(ValueSource::Opening(
            self.accumulator,
        ))])
    }
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<Fr>) -> Vec<Fr> {
        vec![]
    }
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(self.input); if self.square { 2 } else { 1 }],
            ),
        ]))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        vec![self.selector(r)]
    }
}
#[derive(allocative::Allocative)]
struct ReductionProver {
    #[allocative(skip)]
    params: ReductionParams,
    input: MultilinearPolynomial<Fr>,
    selector: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ReductionProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        let half = self.input.len() / 2;
        let values = (0..=self.params.degree())
            .map(|k| {
                let z = Fr::from(k as u64);
                (0..half)
                    .map(|i| {
                        let at = |p: &MultilinearPolynomial<Fr>| {
                            let a = p.get_bound_coeff(i);
                            a + z * (p.get_bound_coeff(i + half) - a)
                        };
                        at(&self.selector) * self.params.value(at(&self.input))
                    })
                    .sum()
            })
            .collect::<Vec<_>>();
        UniPoly::from_evals(&values)
    }
    fn ingest_challenge(&mut self, r: <Fr as JoltField>::Challenge, _: usize) {
        self.input.bind_parallel(r, BindingOrder::HighToLow);
        self.selector.bind_parallel(r, BindingOrder::HighToLow);
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        a.append_dense(t, self.params.input, r.to_vec(), self.input.final_claim());
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, f: &mut allocative::FlameGraphBuilder) {
        f.visit_root(self);
    }
}
struct ReductionVerifier(ReductionParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ReductionVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        self.0.selector(r)
            * self
                .0
                .value(a.get_committed_polynomial_opening(self.0.input).1)
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        a.append_dense(t, self.0.input, r.to_vec());
    }
}

#[derive(Clone)]
pub(super) struct ClampParams {
    pub(super) r: Vec<Fr>,
    pub(super) divisor: Fr,
    pub(super) gamma: [Fr; 3],
    pub(super) openings: Vec<OpeningId>,
}
impl ClampParams {
    fn evaluate(&self, v: &[Fr]) -> Fr {
        let (y, acc, lo, hi) = (v[0], v[1], v[2], v[3]);
        let rem = v.get(4).copied().unwrap_or_default();
        self.gamma[0] * (acc - self.divisor * (y - lo + hi) - rem)
            + self.gamma[1] * lo * (y + Fr::from(1u64 << 31))
            + self.gamma[2] * hi * (Fr::from(i32::MAX as u64) - y)
    }
    fn coefficients(&self) -> Vec<Fr> {
        let mut result = vec![
            self.gamma[0],
            -self.gamma[0] * self.divisor,
            self.gamma[0] * self.divisor,
            -self.gamma[0] * self.divisor,
            self.gamma[1],
            self.gamma[1] * Fr::from(1u64 << 31),
            -self.gamma[2],
            self.gamma[2] * Fr::from(i32::MAX as u64),
        ];
        if self.openings.len() == 5 {
            result.push(-self.gamma[0]);
        }
        result
    }
}
impl SumcheckInstanceParams<Fr> for ClampParams {
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
        let mut factors = vec![
            vec![1],
            vec![0],
            vec![2],
            vec![3],
            vec![2, 0],
            vec![2],
            vec![3, 0],
            vec![3],
        ];
        if self.openings.len() == 5 {
            factors.push(vec![4]);
        }
        Some(OutputClaimConstraint::sum_of_products(
            factors
                .into_iter()
                .enumerate()
                .map(|(i, ids)| {
                    ProductTerm::scaled(
                        ValueSource::Challenge(i),
                        ids.into_iter()
                            .map(|j| ValueSource::Opening(self.openings[j]))
                            .collect(),
                    )
                })
                .collect(),
        ))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        let eq = EqPolynomial::mle(&self.r, r);
        self.coefficients().into_iter().map(|v| eq * v).collect()
    }
}
#[derive(allocative::Allocative)]
pub(super) struct ClampProver {
    #[allocative(skip)]
    pub(super) params: ClampParams,
    pub(super) values: Vec<MultilinearPolynomial<Fr>>,
    pub(super) eq: MultilinearPolynomial<Fr>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ClampProver {
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
                        let values: [Fr; 5] =
                            std::array::from_fn(|j| self.values.get(j).map(at).unwrap_or_default());
                        at(&self.eq) * self.params.evaluate(&values)
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
        for i in 0..self.params.openings.len() {
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
pub(super) struct ClampVerifier(pub(super) ClampParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ClampVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        EqPolynomial::mle(r, &self.0.r)
            * self.0.evaluate(
                &self
                    .0
                    .openings
                    .iter()
                    .map(|id| a.get_committed_polynomial_opening(*id).1)
                    .collect::<Vec<_>>(),
            )
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..self.0.openings.len() {
            a.append_dense(t, self.0.openings[i], r.to_vec());
        }
    }
}
