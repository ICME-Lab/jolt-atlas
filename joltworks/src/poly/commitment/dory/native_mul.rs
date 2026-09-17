//! Native private fused multiplication with floor rescaling and an i32 clamp.
//!
//! For each row, q = floor(a*b / 2^s) and y = clamp_i32(q). All inputs,
//! outputs, remainders and clamp gaps are hidden under Dory commitments.
//! The ranges and three arithmetic identities share one BlindFold argument.

use super::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use crate::{
    curve::{Bn254Curve, Bn254G1},
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
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
        blindfold::{
            assembly::NativeBlindFold,
            protocol::{BlindFoldProof, BlindFoldVerifierInput},
            witness::ExtraConstraintWitness,
            BlindFoldAccumulator, InputClaimConstraint, OutputClaimConstraint, ProductTerm,
            ValueSource,
        },
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
            LinearClaim, LinearTerm,
        },
        sumcheck::{BatchedSumcheck, ZkSumcheckProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::CommittedPoly;
use std::collections::{BTreeMap, BTreeSet};

type ZkProof = ZkSumcheckProof<Fr, Bn254Curve, Blake2bTranscript>;
// Dense tensors, in the fixed order a, b, y, low gap, high gap, remainder.
const N: usize = 6;
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}
fn opening(i: usize, stage: usize) -> OpeningId {
    OpeningId::new(tensor(i), SumcheckId::NodeExecution(stage))
}
fn indicator(i: usize, d: usize) -> CommittedPoly {
    CommittedPoly::NodeOutputRaD(i, d)
}
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeMulStatement {
    pub context: Vec<u8>,
    pub log_rows: usize,
    /// Positive rebase width. The Atlas remainder representation supports 1..=30.
    pub shift: u8,
    pub commitments: BTreeMap<CommittedPoly, DoryCommitment>,
}

/// Private witness state, deliberately not serializable.
pub struct NativeMulWitness {
    polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    hints: BTreeMap<CommittedPoly, DoryHint>,
    range_values: Vec<Vec<u64>>,
    output: Vec<i32>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeMulProof {
    pub relations: ZkProof,
    pub openings: ZkProof,
    pub pcs: DoryProof,
    pub blindfold: BlindFoldProof<Fr, Bn254Curve>,
}

#[derive(Clone, Copy)]
struct Range {
    tensor: usize,
    bits: usize,
    chunk: usize,
    offset: u64,
}
impl Range {
    fn new(tensor: usize, bits: usize, offset: u64) -> Self {
        // Full chunks only: no unconstrained leading padding bits. Short
        // remainder widths use smaller chunks without materializing 2^bits.
        let chunk = [8, 4, 2, 1]
            .into_iter()
            .find(|c| bits.is_multiple_of(*c))
            .unwrap();
        Self {
            tensor,
            bits,
            chunk,
            offset,
        }
    }
    fn chunks(&self) -> usize {
        self.bits / self.chunk
    }
    fn digit(&self, value: u64, d: usize) -> u8 {
        ((value >> (d * self.chunk)) & ((1 << self.chunk) - 1)) as u8
    }
    fn params(&self, r: &[Fr], t: &mut Blake2bTranscript) -> BooleanitySumcheckParams<Fr> {
        let d = self.chunks();
        let hamming: Vec<Fr> = t.challenge_vector(d);
        let beta: Fr = t.challenge_scalar();
        BooleanitySumcheckParams {
            linear: Some(LinearTerm {
                gammas: hamming.clone(),
                weights: (0..d)
                    .map(|i| beta * Fr::from(1u64 << (i * self.chunk)))
                    .collect(),
                claim: LinearClaim {
                    constant: hamming.iter().sum::<Fr>() + beta * Fr::from(self.offset),
                    terms: vec![(opening(self.tensor, 0), beta)],
                },
            }),
            d,
            log_k_chunk: self.chunk,
            log_t: r.len(),
            gammas: t.challenge_vector_optimized::<Fr>(d),
            r_address: t.challenge_vector(self.chunk),
            r_cycle: r.to_vec(),
            polynomial_types: (0..d).map(|j| indicator(self.tensor, j)).collect(),
            sumcheck_id: SumcheckId::Booleanity,
        }
    }
    fn prover(
        &self,
        values: &[u64],
        params: BooleanitySumcheckParams<Fr>,
    ) -> BooleanitySumcheckProver<Fr> {
        let eq = EqPolynomial::<Fr>::evals(&params.r_cycle);
        let mut g = vec![vec![Fr::zero(); 1 << self.chunk]; self.chunks()];
        let mut indices = vec![Vec::with_capacity(values.len()); self.chunks()];
        for (i, value) in values.iter().enumerate() {
            for d in 0..self.chunks() {
                let k = self.digit(*value, d);
                g[d][k as usize] += eq[i];
                indices[d].push(Some(k));
            }
        }
        BooleanitySumcheckProver::gen(params, g, indices)
    }
}

impl NativeMulStatement {
    fn ranges(&self) -> Vec<Range> {
        vec![
            Range::new(0, 32, 1 << 31),
            Range::new(1, 32, 1 << 31),
            Range::new(2, 32, 1 << 31),
            Range::new(3, 64, 0),
            Range::new(4, 64, 0),
            Range::new(5, self.shift as usize, 0),
        ]
    }
    fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        if !(1..=30).contains(&self.shift)
            || self.log_rows.checked_add(8).is_none_or(|n| n > max_vars)
            || self.log_rows >= usize::BITS as usize
        {
            return Err(invalid(
                "Unsupported native multiplication shape or rebase width",
            ));
        }
        let required: BTreeSet<_> = (0..N)
            .map(tensor)
            .chain(
                self.ranges()
                    .iter()
                    .flat_map(|r| (0..r.chunks()).map(move |d| indicator(r.tensor, d))),
            )
            .collect();
        if !required.iter().eq(self.commitments.keys()) {
            return Err(invalid(
                "Every multiplication tensor and range commitment is required",
            ));
        }
        Ok(())
    }
    fn transcript(&self) -> Blake2bTranscript {
        let mut t = Blake2bTranscript::new(b"Atlas/private-fused-mul/v1");
        t.append_serializable(self);
        t
    }
    pub fn left_commitment(&self) -> Option<&DoryCommitment> {
        self.commitments.get(&tensor(0))
    }
    pub fn right_commitment(&self) -> Option<&DoryCommitment> {
        self.commitments.get(&tensor(1))
    }
    pub fn output_commitment(&self) -> Option<&DoryCommitment> {
        self.commitments.get(&tensor(2))
    }
}

impl NativeMulWitness {
    pub fn commit(
        context: Vec<u8>,
        left: &[i32],
        right: &[i32],
        shift: u8,
        setup: &DoryProverSetup,
    ) -> Result<(NativeMulStatement, Self), ProofVerifyError> {
        if left.is_empty()
            || !left.len().is_power_of_two()
            || left.len() != right.len()
            || !(1..=30).contains(&shift)
            || left.len().ilog2() as usize + 8 > setup.verifier.max_log_n
        {
            return Err(invalid("Invalid native multiplication inputs or shape"));
        }
        let divisor = 1i64 << shift;
        let mut output = Vec::with_capacity(left.len());
        let mut lo = Vec::with_capacity(left.len());
        let mut hi = Vec::with_capacity(left.len());
        let mut rem = Vec::with_capacity(left.len());
        for (a, b) in left.iter().zip(right) {
            let product = i64::from(*a) * i64::from(*b);
            let q = product.div_euclid(divisor);
            let y = q.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
            output.push(y as i32);
            lo.push((y - q).max(0) as u64);
            hi.push((q - y).max(0) as u64);
            rem.push(product.rem_euclid(divisor) as u64);
        }
        let signed_offset = |v: &[i32]| {
            v.iter()
                .map(|x| (i64::from(*x) + (1i64 << 31)) as u64)
                .collect::<Vec<_>>()
        };
        let range_values = vec![
            signed_offset(left),
            signed_offset(right),
            signed_offset(&output),
            lo.clone(),
            hi.clone(),
            rem.clone(),
        ];
        let mut polynomials = BTreeMap::from([
            (tensor(0), MultilinearPolynomial::from(left.to_vec())),
            (tensor(1), MultilinearPolynomial::from(right.to_vec())),
            (tensor(2), MultilinearPolynomial::from(output.clone())),
            (tensor(3), MultilinearPolynomial::from(lo)),
            (tensor(4), MultilinearPolynomial::from(hi)),
            (tensor(5), MultilinearPolynomial::from(rem)),
        ]);
        let mut statement = NativeMulStatement {
            context,
            log_rows: left.len().ilog2() as usize,
            shift,
            commitments: BTreeMap::new(),
        };
        for r in statement.ranges() {
            for d in 0..r.chunks() {
                let indices = range_values[r.tensor]
                    .iter()
                    .map(|v| Some(u16::from(r.digit(*v, d))))
                    .collect();
                polynomials.insert(
                    indicator(r.tensor, d),
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        indices,
                        1 << r.chunk,
                    )),
                );
            }
        }
        let mut hints = BTreeMap::new();
        for (id, p) in &polynomials {
            let (c, h) = DoryScheme::commit_zk(p, setup);
            statement.commitments.insert(*id, c);
            hints.insert(*id, h);
        }
        statement.validate(setup.verifier.max_log_n)?;
        Ok((
            statement,
            Self {
                polynomials,
                hints,
                range_values,
                output,
            },
        ))
    }
    /// Prover-side output only. This vector is never a verifier input.
    pub fn output(&self) -> &[i32] {
        &self.output
    }
}

/// Fresh verifier challenges batch the identities
/// a*b = 2^s*(y - low + high) + remainder,
/// low*(y - MIN) = 0, high*(MAX - y) = 0.
/// Signed operand/output ranges and nonnegative gap/remainder ranges make
/// these field identities the integer floor-and-clamp relation. With s<=30,
/// all possible integer differences have magnitude below 2^97, far below Fr.
#[derive(Clone)]
struct ArithmeticParams {
    r: Vec<Fr>,
    divisor: Fr,
    gamma: [Fr; 3],
}
impl ArithmeticParams {
    fn new(r: &[Fr], shift: u8, t: &mut Blake2bTranscript) -> Self {
        Self {
            r: r.to_vec(),
            divisor: Fr::from(1u64 << shift),
            gamma: [
                t.challenge_scalar(),
                t.challenge_scalar(),
                t.challenge_scalar(),
            ],
        }
    }
    fn evaluate(&self, v: &[Fr; N]) -> Fr {
        let [a, b, y, lo, hi, rem] = *v;
        self.gamma[0] * (a * b - self.divisor * (y - lo + hi) - rem)
            + self.gamma[1] * lo * (y + Fr::from(1u64 << 31))
            + self.gamma[2] * hi * (Fr::from(i32::MAX as u64) - y)
    }
    fn coefficients(&self) -> Vec<Fr> {
        vec![
            self.gamma[0],
            -self.gamma[0] * self.divisor,
            self.gamma[0] * self.divisor,
            -self.gamma[0] * self.divisor,
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
            vec![0, 1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
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
                            .map(|j| ValueSource::Opening(opening(j, 1)))
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
            a.append_dense(t, opening(i, 1), r.to_vec(), self.values[i].final_claim());
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
                a.get_committed_polynomial_opening(opening(j, 1)).1
            }))
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        for i in 0..N {
            a.append_dense(t, opening(i, 1), r.to_vec());
        }
    }
}

impl NativeMulProof {
    pub fn prove(
        statement: &NativeMulStatement,
        witness: NativeMulWitness,
        setup: &DoryProverSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<Self, ProofVerifyError> {
        statement.validate(setup.verifier.max_log_n)?;
        super::native_opening::NativeOpeningProof::check_generators(
            &DoryVerifierSetup(setup.verifier.clone()),
            gens,
        )?;
        let NativeMulWitness {
            polynomials,
            hints,
            range_values,
            ..
        } = witness;
        if !polynomials.keys().eq(statement.commitments.keys())
            || !hints.keys().eq(statement.commitments.keys())
            || range_values.len() != N
            || range_values
                .iter()
                .any(|v| v.len() != 1 << statement.log_rows)
        {
            return Err(invalid("Native multiplication witness shape mismatch"));
        }
        let mut t = statement.transcript();
        let r: Vec<Fr> = t.challenge_vector(statement.log_rows);
        let mut a = ProverOpeningAccumulator::new();
        a.zk_mode = true;
        for i in 0..N {
            a.append_dense(
                &mut t,
                opening(i, 0),
                r.clone(),
                polynomials[&tensor(i)].evaluate(&r),
            );
        }
        a.take_pending_claims();
        a.take_pending_claim_ids();
        let arith = ArithmeticProver {
            params: ArithmeticParams::new(&r, statement.shift, &mut t),
            values: std::array::from_fn(|i| polynomials[&tensor(i)].clone()),
            eq: MultilinearPolynomial::from(EqPolynomial::<Fr>::evals(&r)),
        };
        let mut provers: Vec<Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>> =
            vec![Box::new(arith)];
        for range in statement.ranges() {
            let params = range.params(&r, &mut t);
            provers.push(Box::new(range.prover(&range_values[range.tensor], params)));
        }
        let mut bf = BlindFoldAccumulator::new();
        let mut rng = rand::thread_rng();
        let instances = provers
            .iter_mut()
            .map(|p| p.as_mut() as &mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>)
            .collect();
        let (relations, _, _) =
            BatchedSumcheck::prove_zk(instances, &mut a, &mut bf, &mut t, gens, &mut rng);
        a.prepare_for_sumcheck(&polynomials, &mut t);
        let (openings, point) = a.prove_batch_opening_sumcheck_zk::<_, Bn254Curve, _>(
            &mut bf,
            &mut vec![],
            gens,
            &mut rng,
            &mut t,
        );
        let state = a.finalize_batch_opening_sumcheck(point, &mut t);
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let value = coefficients
            .iter()
            .zip(&state.sumcheck_claims)
            .map(|(x, y)| *x * *y)
            .sum();
        let (pcs, eval, blind) = DoryScheme::prove_rlc_zk(
            setup,
            &polynomials,
            &state.poly_coeffs,
            hints.into_values().collect(),
            &state.r_sumcheck,
            &mut t,
        )?;
        if eval != gens.commit(&[value], &blind) {
            return Err(invalid("Multiplication PCS evaluation mismatch"));
        }
        let data = bf.take_stage_data();
        let native = NativeBlindFold::new(
            NativeBlindFold::prover_relations(&data),
            &[relation],
            &coefficients,
            gens.message_generators.len(),
        )?;
        let values = a.openings.iter().map(|(id, (_, v))| (*id, *v)).collect();
        let blindfold = native.prove(
            &data,
            &values,
            vec![ExtraConstraintWitness {
                output_value: value,
                blinding: blind,
                challenge_values: coefficients,
                opening_values: state.sumcheck_claims,
            }],
            vec![eval],
            gens,
            &mut t,
        )?;
        Ok(Self {
            relations,
            openings,
            pcs,
            blindfold,
        })
    }
    pub fn verify(
        &self,
        statement: &NativeMulStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        statement.validate(setup.0.max_log_n)?;
        super::native_opening::NativeOpeningProof::check_generators(setup, gens)?;
        let mut t = statement.transcript();
        let r: Vec<Fr> = t.challenge_vector(statement.log_rows);
        let mut a = VerifierOpeningAccumulator::new_zk();
        for i in 0..N {
            a.append_dense(&mut t, opening(i, 0), r.clone());
        }
        let arith = ArithmeticVerifier(ArithmeticParams::new(&r, statement.shift, &mut t));
        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>> =
            vec![Box::new(arith)];
        for range in statement.ranges() {
            verifiers.push(Box::new(BooleanitySumcheckVerifier::new(
                range.params(&r, &mut t),
            )));
        }
        let instances = verifiers.iter().map(|p| p.as_ref()).collect();
        BatchedSumcheck::verify_zk_with_width(
            &self.relations,
            instances,
            &mut a,
            &mut t,
            gens.message_generators.len(),
        )?;
        let point = a.verify_batch_opening_sumcheck_zk(
            &self.openings,
            &mut t,
            gens.message_generators.len(),
        )?;
        let state =
            a.finalize_batch_opening_sumcheck(point, &vec![Fr::zero(); a.num_sumchecks()], &mut t);
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let commitments = state
            .polynomials
            .iter()
            .map(|p| statement.commitments[p])
            .collect::<Vec<_>>();
        let joint = DoryScheme::combine_commitments(&commitments, &state.poly_coeffs);
        let eval = self
            .pcs
            .0
            .y_com
            .map(|g| Bn254G1(g.0))
            .ok_or_else(|| invalid("Missing hidden multiplication evaluation"))?;
        DoryScheme::verify_zk(&self.pcs, setup, &mut t, &state.r_sumcheck, &eval, &joint)?;
        let native = NativeBlindFold::new(
            a.zk_stages,
            &[relation],
            &coefficients,
            gens.message_generators.len(),
        )?;
        let stages = [&self.relations, &self.openings];
        native.verify(
            &self.blindfold,
            &BlindFoldVerifierInput {
                round_commitments: stages
                    .iter()
                    .flat_map(|s| s.round_commitments.iter().copied())
                    .collect(),
                output_claims_row_commitments: stages
                    .iter()
                    .flat_map(|s| s.output_claims_commitments.iter().copied())
                    .collect(),
                eval_commitments: vec![eval],
            },
            gens,
            &mut t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_onnx_tracer::{
        ops::{Mul, Op},
        tensor::Tensor,
    };

    #[test]
    fn native_mul_matches_atlas_floor_and_clamp_after_serialization() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let a = [i32::MIN, i32::MAX, i32::MIN, -1, 1, 16385, -16385, 0];
        let b = [i32::MIN, i32::MAX, i32::MAX, 1, -1, 16384, 16384, i32::MAX];
        for shift in [1u8, 14, 28, 30] {
            let left = Tensor::new(Some(&a), &[a.len()]).unwrap();
            let right = Tensor::new(Some(&b), &[b.len()]).unwrap();
            let expected = Mul {
                scale: i32::from(shift),
            }
            .f(vec![&left, &right]);
            let (s, w) =
                NativeMulWitness::commit(b"registered-fused-Mul".to_vec(), &a, &b, shift, &pp)
                    .unwrap();
            assert_eq!(w.output(), expected.data());
            assert!(w.output().contains(&-1));
            if shift <= 14 {
                assert!(w.output().contains(&i32::MIN));
                assert!(w.output().contains(&i32::MAX));
            }
            let p = NativeMulProof::prove(&s, w, &pp, &gens).unwrap();
            let mut bytes = vec![];
            p.serialize_compressed(&mut bytes).unwrap();
            let p = NativeMulProof::deserialize_compressed(bytes.as_slice()).unwrap();
            p.verify(&s, &vp, &gens).unwrap();
            println!("native fused Mul shift={shift} proof bytes={}", bytes.len());
            let mut wrong = s.clone();
            wrong.shift = if shift == 14 { 13 } else { 14 };
            assert!(p.verify(&wrong, &vp, &gens).is_err());
            let mut wrong = s.clone();
            wrong.context.push(1);
            assert!(p.verify(&wrong, &vp, &gens).is_err());
            let mut wrong = s.clone();
            wrong.commitments.remove(&indicator(5, 0));
            assert!(p.verify(&wrong, &vp, &gens).is_err());
            let mut wrong = p.clone();
            wrong.pcs.0.y_com = None;
            assert!(wrong.verify(&s, &vp, &gens).is_err());
            let mut wrong = p.clone();
            wrong.relations.output_claims_commitments.clear();
            assert!(wrong.verify(&s, &vp, &gens).is_err());
            let mut wrong = p.clone();
            wrong.blindfold.folded_eval_outputs[0] += Fr::from(1u64);
            assert!(wrong.verify(&s, &vp, &gens).is_err());
        }
    }

    #[test]
    fn native_mul_proves_scalar_inputs_without_clear_claim_fallback() {
        let pp = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let (s, w) = NativeMulWitness::commit(vec![], &[-1], &[1], 14, &pp).unwrap();
        assert_eq!(w.output(), &[-1]);
        NativeMulProof::prove(&s, w, &pp, &gens)
            .unwrap()
            .verify(&s, &vp, &gens)
            .unwrap();
    }

    // Replace both the dense witness and its indicators. Rejection must come
    // from the integer relation, not a stale auxiliary commitment.
    fn replace(
        s: &mut NativeMulStatement,
        w: &mut NativeMulWitness,
        i: usize,
        value: i64,
        pp: &DoryProverSetup,
    ) {
        let range = s.ranges()[i];
        let encoded = (i128::from(value) + i128::from(range.offset)) as u64;
        w.range_values[i] = vec![encoded];
        let p = MultilinearPolynomial::from(vec![value]);
        let (c, h) = DoryScheme::commit_zk(&p, pp);
        s.commitments.insert(tensor(i), c);
        w.hints.insert(tensor(i), h);
        w.polynomials.insert(tensor(i), p);
        for d in 0..range.chunks() {
            let id = indicator(i, d);
            let p = MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                vec![Some(u16::from(range.digit(encoded, d)))],
                1 << range.chunk,
            ));
            let (c, h) = DoryScheme::commit_zk(&p, pp);
            s.commitments.insert(id, c);
            w.hints.insert(id, h);
            w.polynomials.insert(id, p);
        }
    }

    #[test]
    fn native_mul_rejects_wrong_floor_unbounded_remainder_and_false_clamp() {
        let pp = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for attack in 0..7 {
            let (mut s, mut w) = NativeMulWitness::commit(vec![], &[-1], &[1], 14, &pp).unwrap();
            match attack {
                0 => {
                    // Truncation towards zero with a negative remainder.
                    replace(&mut s, &mut w, 2, 0, &pp);
                    replace(&mut s, &mut w, 5, -1, &pp);
                }
                1 => {
                    // One too low with a remainder at least the divisor.
                    replace(&mut s, &mut w, 2, -2, &pp);
                    replace(&mut s, &mut w, 5, (1 << 15) - 1, &pp);
                }
                2 => {
                    // Arithmetic holds and ranges hold, but clamp is false.
                    replace(&mut s, &mut w, 2, 0, &pp);
                    replace(&mut s, &mut w, 3, 1, &pp);
                }
                3 => {
                    // Hiding an out-of-range output with a compensating gap.
                    replace(&mut s, &mut w, 2, i64::from(i32::MAX) + 1, &pp);
                    replace(&mut s, &mut w, 3, i64::from(i32::MAX) + 2, &pp);
                }
                4 => {
                    // A false upper clamp, with the arithmetic relation intact.
                    replace(&mut s, &mut w, 2, -2, &pp);
                    replace(&mut s, &mut w, 4, 1, &pp);
                }
                5 => {
                    // An out-of-range left input with its correct quotient.
                    replace(&mut s, &mut w, 0, 1 << 31, &pp);
                    replace(&mut s, &mut w, 2, 1 << 17, &pp);
                    replace(&mut s, &mut w, 5, 0, &pp);
                }
                _ => {
                    // An out-of-range right input with its correct quotient.
                    replace(&mut s, &mut w, 1, 1 << 31, &pp);
                    replace(&mut s, &mut w, 2, -(1 << 17), &pp);
                    replace(&mut s, &mut w, 5, 0, &pp);
                }
            }
            let p = NativeMulProof::prove(&s, w, &pp, &gens);
            if let Ok(p) = p {
                assert!(p.verify(&s, &vp, &gens).is_err(), "attack {attack}");
            }
        }
        assert!(NativeMulWitness::commit(vec![], &[1], &[1], 0, &pp).is_err());
        assert!(NativeMulWitness::commit(vec![], &[1], &[1], 31, &pp).is_err());
    }
}
