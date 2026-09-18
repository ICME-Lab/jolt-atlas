//! Native exact tensor contractions with hidden operands and bounded integer accumulation.

use super::{
    native_mul::Range,
    native_reduce::{shape_bits, ClampParams, ClampProver, ClampVerifier},
};
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
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};
use ark_bn254::Fr;
#[cfg(test)]
use ark_std::Zero;
use atlas_onnx_tracer::{
    ops::{Einsum, Op},
    tensor::Tensor,
};
use common::{parallel::par_enabled, CommittedPoly};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

type Provers = Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>;
type Verifiers = Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>>;

fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}

#[derive(Clone)]
enum Coordinate {
    Shared(usize),
    Fixed(usize),
}

/// A two-input contraction without repeated labels or nontrivial private
/// summation axes. Shared output axes remain in the sumcheck so batches cannot
/// mix. Public operand bounds ensure every integer partial sum fits in i64.
#[derive(Clone)]
pub(super) struct Contraction {
    pub output_shape: Vec<usize>,
    pub log_inputs: [usize; 2],
    pub log_output: usize,
    pub log_shared: usize,
    pub shift: u8,
    pub input_bits: [u8; 2],
    pub equation: String,
    input_shapes: [Vec<usize>; 2],
    points: [Vec<Coordinate>; 2],
    shared_output: Vec<(usize, usize)>,
}
fn label_bits(labels: &[char], sizes: &BTreeMap<char, usize>) -> BTreeMap<char, usize> {
    let mut start = 0;
    labels
        .iter()
        .map(|c| {
            let offset = start;
            start += sizes[c].ilog2() as usize;
            (*c, offset)
        })
        .collect()
}
fn project(index: usize, positions: &[usize], bits: usize) -> usize {
    positions
        .iter()
        .fold(0, |acc, p| (acc << 1) | ((index >> (bits - 1 - p)) & 1))
}
impl Contraction {
    pub fn new(
        equation: &str,
        shapes: [&[usize]; 2],
        shift: u8,
        input_bits: [u8; 2],
    ) -> Result<Self, ProofVerifyError> {
        let log_inputs = [shape_bits(shapes[0])?, shape_bits(shapes[1])?];
        if shift > 30 || input_bits.iter().any(|b| !(1..=32).contains(b)) {
            return Err(invalid("Invalid contraction scale or operand range"));
        }
        let (args, out) = equation
            .split_once("->")
            .ok_or_else(|| invalid("Contraction requires an explicit output"))?;
        let terms = args.split(',').collect::<Vec<_>>();
        if terms.len() != 2 {
            return Err(invalid("Contraction requires two operands"));
        }
        let labels = [
            terms[0].chars().collect::<Vec<_>>(),
            terms[1].chars().collect::<Vec<_>>(),
        ];
        let output = out.chars().collect::<Vec<_>>();
        for term in labels.iter().chain(std::iter::once(&output)) {
            if term.iter().any(|c| !c.is_ascii_alphabetic())
                || term.iter().collect::<BTreeSet<_>>().len() != term.len()
            {
                return Err(invalid(
                    "Contraction labels must be distinct ASCII letters in each term",
                ));
            }
        }
        let mut sizes = BTreeMap::new();
        for side in 0..2 {
            if labels[side].len() != shapes[side].len() {
                return Err(invalid("Contraction labels do not match operand rank"));
            }
            for (c, size) in labels[side].iter().zip(shapes[side]) {
                if sizes.insert(*c, *size).is_some_and(|old| old != *size) {
                    return Err(invalid("Shared contraction dimensions differ"));
                }
            }
        }
        for c in &output {
            sizes.entry(*c).or_insert(1);
        }
        let output_shape = if output.is_empty() {
            vec![1]
        } else {
            output.iter().map(|c| sizes[c]).collect()
        };
        let log_output = shape_bits(&output_shape)?;
        for side in 0..2 {
            if labels[side]
                .iter()
                .any(|c| !output.contains(c) && !labels[1 - side].contains(c) && sizes[c] != 1)
            {
                return Err(invalid(
                    "Nontrivial contracted axes must occur in both operands",
                ));
            }
        }
        let shared = labels[0]
            .iter()
            .filter(|c| labels[1].contains(c))
            .copied()
            .collect::<Vec<_>>();
        let log_shared = shared.iter().map(|c| sizes[c].ilog2() as usize).sum();
        let log_contract: usize = shared
            .iter()
            .filter(|c| !output.contains(c))
            .map(|c| sizes[c].ilog2() as usize)
            .sum();
        // Each absolute product is <= 2^(a+b-2). Keep their total <= 2^62,
        // so accumulation is exact in i64 regardless of kernel ordering.
        if usize::from(input_bits[0]) + usize::from(input_bits[1]) + log_contract > 64 {
            return Err(invalid(
                "Contraction operand bounds do not exclude integer accumulation overflow",
            ));
        }
        let shared_offsets = label_bits(&shared, &sizes);
        let output_offsets = label_bits(&output, &sizes);
        let points = std::array::from_fn(|side| {
            labels[side]
                .iter()
                .flat_map(|c| {
                    (0..sizes[c].ilog2() as usize).map(|b| {
                        if let Some(start) = shared_offsets.get(c) {
                            Coordinate::Shared(start + b)
                        } else {
                            Coordinate::Fixed(output_offsets[c] + b)
                        }
                    })
                })
                .collect()
        });
        let shared_output = shared
            .iter()
            .filter(|c| output.contains(c))
            .flat_map(|c| {
                (0..sizes[c].ilog2() as usize)
                    .map(|b| (shared_offsets[c] + b, output_offsets[c] + b))
            })
            .collect();
        Ok(Self {
            output_shape,
            log_inputs,
            log_output,
            log_shared,
            shift,
            input_bits,
            equation: equation.into(),
            input_shapes: shapes.map(|s| s.to_vec()),
            points,
            shared_output,
        })
    }
    pub fn columns(&self) -> usize {
        if self.shift == 0 {
            6
        } else {
            7
        }
    }
    pub fn ranges(&self) -> Vec<Range> {
        let mut ranges = vec![
            Range::new(
                0,
                usize::from(self.input_bits[0]),
                1u64 << (self.input_bits[0] - 1),
            ),
            Range::new(
                1,
                usize::from(self.input_bits[1]),
                1u64 << (self.input_bits[1] - 1),
            ),
            Range::new(2, 32, 1 << 31),
            Range::new(3, 64, 1 << 63),
            Range::new(4, 64, 0),
            Range::new(5, 64, 0),
        ];
        if self.shift > 0 {
            ranges.push(Range::new(6, usize::from(self.shift), 0));
        }
        ranges
    }
    /// Fix private output coordinates. This table has only shared axes,
    /// never the Cartesian product of all output and contraction axes.
    fn partial(
        &self,
        side: usize,
        poly: &MultilinearPolynomial<Fr>,
        r: &[Fr],
    ) -> MultilinearPolynomial<Fr> {
        self.partial_with_scalars(side, poly, r, true)
    }
    fn partial_with_scalars(
        &self,
        side: usize,
        poly: &MultilinearPolynomial<Fr>,
        r: &[Fr],
        use_small_scalars: bool,
    ) -> MultilinearPolynomial<Fr> {
        // Original integer inputs can use the existing small-scalar field
        // multiplication. Bound polynomials must use their current field values.
        let compact = match poly {
            MultilinearPolynomial::I32Scalars(p) if use_small_scalars && !p.is_bound() => {
                Some(p.coeffs.as_slice())
            }
            _ => None,
        };
        let fixed = self.points[side]
            .iter()
            .filter_map(|p| match p {
                Coordinate::Fixed(i) => Some(r[*i]),
                _ => None,
            })
            .collect::<Vec<_>>();
        let weights = EqPolynomial::<Fr>::evals(&fixed);
        let positions = self.points[side]
            .iter()
            .enumerate()
            .filter_map(|(i, p)| matches!(p, Coordinate::Fixed(_)).then_some(i))
            .collect::<Vec<_>>();
        // Split the index map into two small tables. Their disjoint bits can
        // be combined with OR, avoiding a coordinate loop per coefficient.
        // No worker needs a separate copy of the output table.
        let low_bits = positions.len().div_ceil(2);
        let split = positions.len() - low_bits;
        let offsets = |positions: &[usize]| {
            (0..1usize << positions.len())
                .map(|value| {
                    positions
                        .iter()
                        .enumerate()
                        .fold(0usize, |index, (bit, position)| {
                            index
                                | (((value >> (positions.len() - 1 - bit)) & 1)
                                    << (self.log_inputs[side] - 1 - position))
                        })
                })
                .collect::<Vec<_>>()
        };
        let high = offsets(&positions[..split]);
        let low = offsets(&positions[split..]);
        let low_mask = low.len() - 1;
        let table = (0..1usize << self.log_shared)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|shared| {
                let base = self.points[side].iter().enumerate().fold(
                    0usize,
                    |index, (bit, coordinate)| match coordinate {
                        Coordinate::Shared(j) => {
                            index
                                | (((shared >> (self.log_shared - 1 - j)) & 1)
                                    << (self.log_inputs[side] - 1 - bit))
                        }
                        Coordinate::Fixed(_) => index,
                    },
                );
                weights
                    .iter()
                    .enumerate()
                    .map(|(fixed, weight)| {
                        let index = base | high[fixed >> low_bits] | low[fixed & low_mask];
                        compact.map_or_else(
                            || poly.get_bound_coeff(index) * *weight,
                            |coefficients| coefficients[index].field_mul(*weight),
                        )
                    })
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();
        MultilinearPolynomial::from(table)
    }
    fn selector_table(&self, r: &[Fr]) -> MultilinearPolynomial<Fr> {
        let coordinates = self
            .shared_output
            .iter()
            .map(|(_, out)| r[*out])
            .collect::<Vec<_>>();
        let weights = EqPolynomial::<Fr>::evals(&coordinates);
        let positions = self
            .shared_output
            .iter()
            .map(|(bit, _)| *bit)
            .collect::<Vec<_>>();
        MultilinearPolynomial::from(
            (0..1 << self.log_shared)
                .map(|i| weights[project(i, &positions, self.log_shared)])
                .collect::<Vec<_>>(),
        )
    }
    fn selector(&self, output: &[Fr], shared: &[<Fr as JoltField>::Challenge]) -> Fr {
        EqPolynomial::mle(
            &self
                .shared_output
                .iter()
                .map(|(_, out)| output[*out])
                .collect::<Vec<_>>(),
            &self
                .shared_output
                .iter()
                .map(|(bit, _)| shared[*bit])
                .collect::<Vec<_>>(),
        )
    }
    fn opening_point(
        &self,
        side: usize,
        output: &[Fr],
        shared: &[<Fr as JoltField>::Challenge],
    ) -> Vec<Fr> {
        self.points[side]
            .iter()
            .map(|p| match p {
                Coordinate::Shared(i) => shared[*i].into(),
                Coordinate::Fixed(i) => output[*i],
            })
            .collect()
    }
}

pub(super) struct ContractionWitness {
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
    pub output: Vec<i32>,
}
impl ContractionWitness {
    pub fn new(inputs: [&[i32]; 2], c: &Contraction) -> Result<Self, ProofVerifyError> {
        for side in 0..2 {
            let limit = 1i64 << (c.input_bits[side] - 1);
            if inputs[side].len() != 1 << c.log_inputs[side]
                || inputs[side]
                    .iter()
                    .any(|x| i64::from(*x) < -limit || i64::from(*x) >= limit)
            {
                return Err(invalid(
                    "Contraction input violates its registered shape or integer range",
                ));
            }
        }
        // Use Atlas's existing blocked integer kernel, retaining its captured
        // quotient and remainder instead of executing the matrix product again.
        let a = Tensor::new(Some(inputs[0]), &c.input_shapes[0])
            .map_err(|_| invalid("Invalid left tensor"))?;
        let b = Tensor::new(Some(inputs[1]), &c.input_shapes[1])
            .map_err(|_| invalid("Invalid right tensor"))?;
        let (out, intermediates) = Einsum {
            equation: c.equation.clone(),
            scale: i32::from(c.shift),
        }
        .f_with_intermediates(vec![&a, &b]);
        let ints =
            intermediates.ok_or_else(|| invalid("Missing exact contraction intermediates"))?;
        if out.dims() != c.output_shape.as_slice() {
            return Err(invalid("Atlas contraction output shape mismatch"));
        }
        let output = out.data().to_vec();
        let mut acc = vec![];
        let mut lo = vec![];
        let mut hi = vec![];
        let mut rem = vec![];
        for ((q, remainder), y) in ints
            .quotient
            .data()
            .iter()
            .zip(ints.remainder.data())
            .zip(&output)
        {
            let exact = i128::from(*q) * (1i128 << c.shift) + i128::from(*remainder);
            acc.push(
                i64::try_from(exact).map_err(|_| invalid("Contraction accumulator overflow"))?,
            );
            lo.push((i128::from(*y) - i128::from(*q)).max(0) as u64);
            hi.push((i128::from(*q) - i128::from(*y)).max(0) as u64);
            rem.push(*remainder as u64);
        }
        let signed = |v: &[i32], bits: u8| {
            v.iter()
                .map(|x| (i64::from(*x) + (1i64 << (bits - 1))) as u64)
                .collect::<Vec<_>>()
        };
        let mut range_values = vec![
            signed(inputs[0], c.input_bits[0]),
            signed(inputs[1], c.input_bits[1]),
            signed(&output, 32),
            acc.iter()
                .map(|x| (i128::from(*x) + (1i128 << 63)) as u64)
                .collect(),
            lo.clone(),
            hi.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(inputs[0].to_vec())),
            (tensor(1), MultilinearPolynomial::from(inputs[1].to_vec())),
            (tensor(2), MultilinearPolynomial::from(output.clone())),
            (
                tensor(3),
                MultilinearPolynomial::from(
                    acc.iter().map(|x| Fr::from_i64(*x)).collect::<Vec<_>>(),
                ),
            ),
            (tensor(4), MultilinearPolynomial::from(lo)),
            (tensor(5), MultilinearPolynomial::from(hi)),
        ]);
        if c.shift > 0 {
            range_values.push(rem.clone());
            polynomials.insert(tensor(6), MultilinearPolynomial::from(rem));
        }
        for r in c.ranges() {
            for d in 0..r.chunks() {
                let digits = range_values[r.tensor]
                    .iter()
                    .map(|v| Some(u16::from(r.digit(*v, d))))
                    .collect();
                polynomials.insert(
                    CommittedPoly::NodeOutputRaD(r.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        digits,
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

pub(super) struct ContractionRegistration {
    /// Left, right, output, accumulator, lower gap, upper gap, optional remainder.
    pub tensors: [CommittedPoly; 7],
    pub initial: [SumcheckId; 3],
    pub product_stages: [SumcheckId; 2],
    pub clamp_stage: SumcheckId,
    pub namespace: usize,
    pub contraction: Contraction,
}
impl ContractionRegistration {
    fn source(&self, i: usize) -> OpeningId {
        OpeningId::new(self.tensors[i], self.initial[i.min(2)])
    }
    pub(super) fn ranges(&self) -> Vec<Range> {
        self.contraction
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
        self.tensors[..self.contraction.columns()]
            .iter()
            .copied()
            .chain(self.ranges().into_iter().flat_map(|r| {
                (0..r.chunks()).map(move |d| CommittedPoly::NodeOutputRaD(r.namespace, d))
            }))
            .collect()
    }
    fn params(&self, r: &[Fr]) -> ContractionParams {
        ContractionParams {
            contraction: self.contraction.clone(),
            r_output: r.to_vec(),
            accumulator: self.source(3),
            inputs: std::array::from_fn(|i| {
                OpeningId::new(self.tensors[i], self.product_stages[i])
            }),
        }
    }
    fn clamp_params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> ClampParams {
        ClampParams {
            r: r.to_vec(),
            divisor: Fr::from(1u64 << self.contraction.shift),
            gamma: std::array::from_fn(|_| t.challenge_scalar()),
            openings: self.tensors[2..self.contraction.columns()]
                .iter()
                .map(|p| OpeningId::new(*p, self.clamp_stage))
                .collect(),
        }
    }
    pub fn provers(
        &self,
        values: &[Vec<u64>],
        polys: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Provers, ProofVerifyError> {
        let c = &self.contraction;
        if values.len() != c.columns()
            || values
                .iter()
                .enumerate()
                .any(|(i, v)| v.len() != 1 << if i < 2 { c.log_inputs[i] } else { c.log_output })
        {
            return Err(invalid("Contraction witness shape mismatch"));
        }
        let points = [
            t.challenge_vector(c.log_inputs[0]),
            t.challenge_vector(c.log_inputs[1]),
            t.challenge_vector(c.log_output),
        ];
        for i in 0..c.columns() {
            let r = &points[i.min(2)];
            a.append_dense(
                t,
                self.source(i),
                r.clone(),
                polys[&self.tensors[i]].evaluate(r),
            );
        }
        let r = &points[2];
        let product = ContractionProver {
            params: self.params(r),
            inputs: std::array::from_fn(|i| c.partial(i, &polys[&self.tensors[i]], r)),
            selector: if c.shared_output.is_empty() {
                None
            } else {
                Some(c.selector_table(r))
            },
        };
        let clamp = ClampProver {
            params: self.clamp_params(r, t),
            values: self.tensors[2..c.columns()]
                .iter()
                .map(|id| polys[id].clone())
                .collect(),
            eq: MultilinearPolynomial::from(EqPolynomial::<Fr>::evals(r)),
        };
        let mut result: Provers = vec![Box::new(product), Box::new(clamp)];
        for range in self.ranges() {
            let params = range.params(&points[range.tensor.min(2)], t);
            result.push(Box::new(range.prover(&values[range.tensor], params)));
        }
        Ok(result)
    }
    pub fn verifiers(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Verifiers {
        let c = &self.contraction;
        let points = [
            t.challenge_vector(c.log_inputs[0]),
            t.challenge_vector(c.log_inputs[1]),
            t.challenge_vector(c.log_output),
        ];
        for i in 0..c.columns() {
            a.append_dense(t, self.source(i), points[i.min(2)].clone());
        }
        let mut result: Verifiers = vec![
            Box::new(ContractionVerifier(self.params(&points[2]))),
            Box::new(ClampVerifier(self.clamp_params(&points[2], t))),
        ];
        for range in self.ranges() {
            result.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&points[range.tensor.min(2)], t),
            )));
        }
        result
    }
}
#[derive(Clone)]
struct ContractionParams {
    contraction: Contraction,
    r_output: Vec<Fr>,
    accumulator: OpeningId,
    inputs: [OpeningId; 2],
}
impl SumcheckInstanceParams<Fr> for ContractionParams {
    fn degree(&self) -> usize {
        if self.contraction.shared_output.is_empty() {
            2
        } else {
            3
        }
    }
    fn num_rounds(&self) -> usize {
        self.contraction.log_shared
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
                self.inputs
                    .iter()
                    .map(|id| ValueSource::Opening(*id))
                    .collect(),
            ),
        ]))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        vec![self.contraction.selector(&self.r_output, r)]
    }
}
#[derive(allocative::Allocative)]
struct ContractionProver {
    #[allocative(skip)]
    params: ContractionParams,
    inputs: [MultilinearPolynomial<Fr>; 2],
    selector: Option<MultilinearPolynomial<Fr>>,
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ContractionProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        let half = self.inputs[0].len() / 2;
        let values = (0..=self.params.degree())
            .map(|k| {
                let z = Fr::from(k as u64);
                (0..half)
                    .map(|i| {
                        let at = |p: &MultilinearPolynomial<Fr>| {
                            let a = p.get_bound_coeff(i);
                            a + z * (p.get_bound_coeff(i + half) - a)
                        };
                        self.selector
                            .as_ref()
                            .map(at)
                            .unwrap_or_else(|| Fr::from(1u64))
                            * at(&self.inputs[0])
                            * at(&self.inputs[1])
                    })
                    .sum()
            })
            .collect::<Vec<_>>();
        UniPoly::from_evals(&values)
    }
    fn ingest_challenge(&mut self, r: <Fr as JoltField>::Challenge, _: usize) {
        if let Some(selector) = &mut self.selector {
            selector.bind_parallel(r, BindingOrder::HighToLow);
        }
        for p in &mut self.inputs {
            p.bind_parallel(r, BindingOrder::HighToLow);
        }
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..2 {
            a.append_dense(
                t,
                self.params.inputs[i],
                self.params
                    .contraction
                    .opening_point(i, &self.params.r_output, r),
                self.inputs[i].final_claim(),
            );
        }
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, f: &mut allocative::FlameGraphBuilder) {
        f.visit_root(self);
    }
}
struct ContractionVerifier(ContractionParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ContractionVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        self.0.contraction.selector(&self.0.r_output, r)
            * self
                .0
                .inputs
                .iter()
                .map(|id| a.get_committed_polynomial_opening(*id).1)
                .product::<Fr>()
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..2 {
            a.append_dense(
                t,
                self.0.inputs[i],
                self.0.contraction.opening_point(i, &self.0.r_output, r),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_einsum_partial_tables_match_full_openings_without_product_expansion() {
        for (eq, shapes, shared_values) in [
            ("mk,kn->mn", [vec![64, 32], vec![32, 128]], 32),
            ("bmk,kbn->nbm", [vec![2, 4, 8], vec![8, 2, 16]], 16),
            ("m,n->nm", [vec![4], vec![8]], 1),
        ] {
            let c = Contraction::new(eq, [&shapes[0], &shapes[1]], 1, [16, 16]).unwrap();
            let mut t = Blake2bTranscript::new(b"partial matrix binding reference");
            let r: Vec<Fr> = t.challenge_vector(c.log_output);
            let z = t.challenge_vector_optimized::<Fr>(c.log_shared);
            let z_field = z.iter().map(|x| (*x).into()).collect::<Vec<Fr>>();
            for side in 0..2 {
                let data = (0..1 << c.log_inputs[side])
                    .map(|i| (i % 17) - 8)
                    .collect::<Vec<_>>();
                let p = MultilinearPolynomial::from(data);
                let partial = c.partial(side, &p, &r);
                assert_eq!(partial.len(), shared_values);
                assert_eq!(
                    partial.evaluate(&z_field),
                    p.evaluate(&c.opening_point(side, &r, &z)),
                    "{eq} operand {side}"
                );
            }
        }
    }
}

#[cfg(test)]
impl Contraction {
    fn serial_partial(
        &self,
        side: usize,
        poly: &MultilinearPolynomial<Fr>,
        r: &[Fr],
    ) -> MultilinearPolynomial<Fr> {
        let fixed = self.points[side]
            .iter()
            .filter_map(|p| match p {
                Coordinate::Fixed(i) => Some(r[*i]),
                _ => None,
            })
            .collect::<Vec<_>>();
        let weights = EqPolynomial::<Fr>::evals(&fixed);
        let positions = self.points[side]
            .iter()
            .enumerate()
            .filter_map(|(i, p)| matches!(p, Coordinate::Fixed(_)).then_some(i))
            .collect::<Vec<_>>();
        let mut table = vec![Fr::zero(); 1 << self.log_shared];
        for i in 0..poly.len() {
            let mut shared = 0;
            for (bit, p) in self.points[side].iter().enumerate() {
                if let Coordinate::Shared(j) = p {
                    shared |=
                        ((i >> (self.log_inputs[side] - 1 - bit)) & 1) << (self.log_shared - 1 - j);
                }
            }
            table[shared] +=
                poly.get_bound_coeff(i) * weights[project(i, &positions, self.log_inputs[side])];
        }
        MultilinearPolynomial::from(table)
    }
}

#[cfg(test)]
mod partial_index_tests {
    use super::*;

    #[test]
    fn native_contraction_partial_indices_match_every_serial_coefficient() {
        for workers in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for (equation, shapes) in [
                        ("mk,kn->mn", [vec![64, 32], vec![32, 128]]),
                        ("bmk,kbn->nbm", [vec![2, 16, 32], vec![32, 2, 128]]),
                        ("m,n->nm", [vec![64], vec![128]]),
                        ("km,nk->mn", [vec![32, 16], vec![128, 32]]),
                        ("ab,ab->", [vec![64, 128], vec![64, 128]]),
                        ("amk,kan->nam", [vec![2, 4, 64], vec![64, 2, 128]]),
                        ("ab,bc->ca", [vec![256, 128], vec![128, 64]]),
                        ("u,uv->v", [vec![1], vec![1, 8192]]),
                        ("ab,b->a", [vec![4096, 16], vec![16]]),
                    ] {
                        let c = Contraction::new(equation, [&shapes[0], &shapes[1]], 1, [16, 16])
                            .unwrap();
                        let mut transcript = Blake2bTranscript::new(b"partial index maps");
                        for point in [
                            transcript.challenge_vector::<Fr>(c.log_output),
                            (0..c.log_output)
                                .map(|i| Fr::from((i % 2) as u64))
                                .collect(),
                        ] {
                            for side in 0..2 {
                                let p = MultilinearPolynomial::from(
                                    (0..1usize << c.log_inputs[side])
                                        .map(|i| ((i * 73 + i / 17) % 1021) as i32 - 511)
                                        .collect::<Vec<_>>(),
                                );
                                let expected = c.serial_partial(side, &p, &point);
                                let actual = c.partial(side, &p, &point);
                                assert_eq!(actual.len(), expected.len());
                                for i in 0..actual.len() {
                                    assert_eq!(
                                        actual.get_bound_coeff(i),
                                        expected.get_bound_coeff(i),
                                        "{equation}, side {side}, coefficient {i}"
                                    );
                                }
                                let _guard = common::parallel::ParallelFlagGuard::disabled();
                                let disabled = c.partial(side, &p, &point);
                                for i in 0..actual.len() {
                                    assert_eq!(
                                        disabled.get_bound_coeff(i),
                                        expected.get_bound_coeff(i)
                                    );
                                }
                            }
                        }
                    }
                });
        }
    }

    #[test]
    fn native_contraction_scaled_tables_preserve_bound_values_and_integer_extremes() {
        let c = Contraction::new("ab,b->a", [&[4, 4], &[4]], 1, [16, 16]).unwrap();
        let mut transcript = Blake2bTranscript::new(b"partial scalar representations");
        let point = transcript.challenge_vector::<Fr>(c.log_output);
        let values: Vec<i32> = (0..16)
            .map(|i| [i32::MIN, -1, 0, 1, i32::MAX][i % 5])
            .collect();
        let original = MultilinearPolynomial::from(values.clone());
        let field = MultilinearPolynomial::from(
            values.iter().map(|v| Fr::from_i32(*v)).collect::<Vec<_>>(),
        );
        let boolean = MultilinearPolynomial::from((0..16).map(|i| i % 3 == 0).collect::<Vec<_>>());
        let mut bound = MultilinearPolynomial::from(
            [values.clone(), values.iter().rev().copied().collect()].concat(),
        );
        bound.bind_parallel(
            transcript.challenge_scalar_optimized::<Fr>(),
            BindingOrder::HighToLow,
        );
        for p in [original, field, boolean, bound] {
            let expected = c.serial_partial(0, &p, &point);
            for candidate in [
                c.partial(0, &p, &point),
                c.partial_with_scalars(0, &p, &point, false),
            ] {
                assert_eq!(candidate.len(), expected.len());
                for i in 0..expected.len() {
                    assert_eq!(candidate.get_bound_coeff(i), expected.get_bound_coeff(i));
                }
            }
        }
    }

    #[test]
    #[ignore = "Isolated contraction table construction, not a complete native proof"]
    fn native_contraction_partial_benchmark() {
        let mode = std::env::var("NATIVE_PARTIAL_MODE").unwrap();
        assert!(mode == "serial" || mode == "indexed" || mode == "scaled");
        let c =
            Contraction::new("ab,b->a", [&[1 << 14, 1 << 10], &[1 << 10]], 1, [16, 16]).unwrap();
        let mut transcript = Blake2bTranscript::new(b"partial table benchmark");
        let point = transcript.challenge_vector::<Fr>(c.log_output);
        let p = MultilinearPolynomial::from(
            (0..1usize << c.log_inputs[0])
                .map(|i| ((i * 73 + i / 17) % 1021) as i32 - 511)
                .collect::<Vec<_>>(),
        );
        let started = std::time::Instant::now();
        let result = match mode.as_str() {
            "serial" => c.serial_partial(0, &p, &point),
            "indexed" => c.partial_with_scalars(0, &p, &point, false),
            _ => c.partial(0, &p, &point),
        };
        let seconds = started.elapsed().as_secs_f64();
        println!("PARTIAL_BENCH {{\"mode\":\"{}\",\"log_rows\":24,\"seconds\":{},\"workers\":{},\"complete_proof\":false}}",
            mode, seconds, rayon::current_num_threads());
        std::hint::black_box(result);
    }
}
