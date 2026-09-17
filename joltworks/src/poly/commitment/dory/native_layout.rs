//! Prove registered broadcast, reshape and axis permutation on hidden tensors.
//! Output evaluation points determine original-input points. A required signed
//! input range prevents a layout-only graph from accepting noninteger tensors.

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
use std::collections::{BTreeMap, BTreeSet};

fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}

#[derive(Clone)]
pub(super) struct Layout {
    pub output_shape: Vec<usize>,
    pub input_bits: Vec<usize>,
    pub log_input: usize,
    pub log_output: usize,
}
impl Layout {
    /// kind 0 broadcasts, 1 preserves flattened order, 2 permutes tensor axes.
    pub fn new(
        input: &[usize],
        kind: u8,
        shape: &[usize],
        axes: &[usize],
    ) -> Result<Self, ProofVerifyError> {
        let log_input = shape_bits(input)?;
        let (output_shape, input_bits) = match kind {
            0 if axes.is_empty() && input.len() <= shape.len() => {
                shape_bits(shape)?;
                let offset = shape.len() - input.len();
                let mut start = 0;
                let mut bits = vec![];
                for (axis, size) in shape.iter().enumerate() {
                    let width = size.ilog2() as usize;
                    if axis >= offset {
                        let source = input[axis - offset];
                        if source != 1 && source != *size {
                            return Err(invalid("Incompatible broadcast dimensions"));
                        }
                        if source != 1 {
                            bits.extend(start..start + width);
                        }
                    }
                    start += width;
                }
                (shape.to_vec(), bits)
            }
            1 if axes.is_empty() && shape_bits(shape)? == log_input => {
                (shape.to_vec(), (0..log_input).collect())
            }
            2 if shape.is_empty()
                && axes.len() == input.len()
                && axes.iter().all(|a| *a < input.len())
                && axes.iter().copied().collect::<BTreeSet<_>>().len() == input.len() =>
            {
                let output = axes.iter().map(|a| input[*a]).collect::<Vec<_>>();
                let mut starts = vec![0; input.len()];
                let mut start = 0;
                for axis in axes {
                    starts[*axis] = start;
                    start += input[*axis].ilog2() as usize;
                }
                let bits = input
                    .iter()
                    .enumerate()
                    .flat_map(|(axis, size)| starts[axis]..starts[axis] + size.ilog2() as usize)
                    .collect();
                (output, bits)
            }
            _ => return Err(invalid("Invalid registered tensor layout")),
        };
        let log_output = shape_bits(&output_shape)?;
        assert_eq!(input_bits.len(), log_input);
        Ok(Self {
            output_shape,
            input_bits,
            log_input,
            log_output,
        })
    }
    pub fn input_index(&self, output: usize) -> usize {
        self.input_bits.iter().fold(0, |index, bit| {
            (index << 1) | ((output >> (self.log_output - 1 - bit)) & 1)
        })
    }
    fn input_point(&self, output: &[Fr]) -> Vec<Fr> {
        self.input_bits.iter().map(|bit| output[*bit]).collect()
    }
}

pub(super) struct LayoutWitness {
    pub output: Vec<i32>,
    pub polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    pub range_values: Vec<Vec<u64>>,
}
impl LayoutWitness {
    pub fn new(input: &[i32], layout: &Layout) -> Result<Self, ProofVerifyError> {
        if input.len() != 1 << layout.log_input {
            return Err(invalid("Layout input size mismatch"));
        }
        let output = (0..1 << layout.log_output)
            .map(|i| input[layout.input_index(i)])
            .collect::<Vec<_>>();
        let values = input
            .iter()
            .map(|x| (i64::from(*x) + (1i64 << 31)) as u64)
            .collect::<Vec<_>>();
        let mut polynomials =
            BTreeMap::from([(tensor(1), MultilinearPolynomial::from(output.clone()))]);
        let r = Range::new(0, 32, 1 << 31);
        for d in 0..r.chunks() {
            polynomials.insert(
                CommittedPoly::NodeOutputRaD(0, d),
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    values
                        .iter()
                        .map(|v| Some(u16::from(r.digit(*v, d))))
                        .collect(),
                    1 << r.chunk,
                )),
            );
        }
        Ok(Self {
            output,
            polynomials,
            range_values: vec![values],
        })
    }
}

pub(super) struct LayoutRegistration {
    pub tensors: [CommittedPoly; 2],
    pub initial: SumcheckId,
    pub final_stage: SumcheckId,
    pub namespace: usize,
    pub layout: Layout,
}
impl LayoutRegistration {
    fn range(&self) -> Range {
        let mut r = Range::new(0, 32, 1 << 31);
        r.namespace = self.namespace;
        r.input = OpeningId::new(self.tensors[0], self.initial);
        r
    }
    pub fn indicator_keys(&self) -> Vec<CommittedPoly> {
        let r = self.range();
        (0..r.chunks())
            .map(|d| CommittedPoly::NodeOutputRaD(r.namespace, d))
            .collect()
    }
    fn params(&self, t: &mut Blake2bTranscript) -> LayoutParams {
        let output: Vec<Fr> = t.challenge_vector(self.layout.log_output);
        LayoutParams {
            openings: self.tensors.map(|id| OpeningId::new(id, self.final_stage)),
            points: [self.layout.input_point(&output), output],
        }
    }
    pub fn provers(
        &self,
        values: &[Vec<u64>],
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Result<Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>>, ProofVerifyError> {
        if values.len() != 1 || values[0].len() != 1 << self.layout.log_input {
            return Err(invalid("Invalid layout range witness"));
        }
        let params = self.params(t);
        let evaluations =
            std::array::from_fn(|j| polynomials[&self.tensors[j]].evaluate(&params.points[j]));
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        let range = self.range();
        a.append_dense(
            t,
            range.input,
            r.clone(),
            polynomials[&self.tensors[0]].evaluate(&r),
        );
        let range_params = range.params(&r, t);
        Ok(vec![
            Box::new(LayoutProver {
                params,
                evaluations,
            }),
            Box::new(range.prover(&values[0], range_params)),
        ])
    }
    pub fn verifiers(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> {
        let params = self.params(t);
        let r: Vec<Fr> = t.challenge_vector(self.layout.log_input);
        let range = self.range();
        a.append_dense(t, range.input, r.clone());
        vec![
            Box::new(LayoutVerifier(params)),
            Box::new(BooleanitySumcheckVerifier::new(range.params(&r, t))),
        ]
    }
}

#[derive(Clone)]
struct LayoutParams {
    openings: [OpeningId; 2],
    points: [Vec<Fr>; 2],
}
impl SumcheckInstanceParams<Fr> for LayoutParams {
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
            (0..2)
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
        vec![Fr::from(1u64), -Fr::from(1u64)]
    }
}
#[derive(allocative::Allocative)]
struct LayoutProver {
    #[allocative(skip)]
    params: LayoutParams,
    #[allocative(skip)]
    evaluations: [Fr; 2],
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for LayoutProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, _: Fr) -> UniPoly<Fr> {
        unreachable!("Layout equality has no sumcheck rounds")
    }
    fn ingest_challenge(&mut self, _: <Fr as JoltField>::Challenge, _: usize) {
        unreachable!("Layout equality has no sumcheck rounds")
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..2 {
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
struct LayoutVerifier(LayoutParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for LayoutVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        _: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        a.get_committed_polynomial_opening(self.0.openings[0]).1
            - a.get_committed_polynomial_opening(self.0.openings[1]).1
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        _: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..2 {
            a.append_dense(t, self.0.openings[i], self.0.points[i].clone());
        }
    }
}
