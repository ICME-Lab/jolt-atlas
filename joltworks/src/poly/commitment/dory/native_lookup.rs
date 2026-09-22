//! Native hidden table lookup for all rows of two registered tensor commitments.
//! The table and tensor shape are public. Input indices, outputs, indicator
//! polynomials and every scalar opening stay hidden. This component does not
//! establish ONNX execution, token generation or cross-encoding equality.

use super::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use crate::{
    config::{OneHotConfig, OneHotParams},
    curve::{Bn254Curve, Bn254G1},
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::{
        blindfold::{
            assembly::NativeBlindFold,
            protocol::{BlindFoldProof, BlindFoldVerifierInput},
            witness::ExtraConstraintWitness,
            BlindFoldAccumulator,
        },
        shout::{self, RaOneHotEncoding, ReadRafProvider},
        sumcheck::{BatchedSumcheck, ZkSumcheckProof},
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::{CommittedPoly, VirtualPoly};
use std::collections::{BTreeMap, BTreeSet};

type SumcheckProof = ZkSumcheckProof<Fr, Bn254Curve, Blake2bTranscript>;
const INPUT: CommittedPoly = CommittedPoly::DivNodeQuotient(0);
const OUTPUT: CommittedPoly = CommittedPoly::DivNodeQuotient(1);
const SOURCE: SumcheckId = SumcheckId::NodeExecution(0);
fn input_id() -> OpeningId {
    OpeningId::new(INPUT, SOURCE)
}
fn output_id() -> OpeningId {
    OpeningId::new(OUTPUT, SOURCE)
}
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}

/// The caller supplies the exact expected statement, including the commitments
/// used by adjacent components. The proof cannot select a different statement.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeLookupStatement {
    pub context: Vec<u8>,
    pub table: Vec<i32>,
    pub log_rows: usize,
    pub log_chunk: u8,
    pub commitments: BTreeMap<CommittedPoly, DoryCommitment>,
}

/// Private witness state. Blinds and indices are deliberately not serializable.
pub struct NativeLookupWitness {
    indices: Vec<usize>,
    polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    hints: BTreeMap<CommittedPoly, DoryHint>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeLookupProof {
    pub read: SumcheckProof,
    pub indicators: SumcheckProof,
    pub openings: SumcheckProof,
    pub pcs: DoryProof,
    pub blindfold: BlindFoldProof<Fr, Bn254Curve>,
}

#[derive(Clone)]
pub(super) struct Lookup {
    pub params: OneHotParams,
    pub log_k: usize,
    pub input: OpeningId,
    pub output: OpeningId,
    pub namespace: usize,
}
impl ReadRafProvider<Fr> for Lookup {
    fn rv_claim(&self, a: &dyn OpeningAccumulator<Fr>) -> Fr {
        a.get_committed_polynomial_opening(self.output).1
    }
    fn raf_claim(&self, a: &dyn OpeningAccumulator<Fr>) -> Fr {
        a.get_committed_polynomial_opening(self.input).1
    }
    fn rv_claim_source(&self) -> OpeningId {
        self.output
    }
    fn raf_claim_source(&self) -> OpeningId {
        self.input
    }
    fn r(&self, a: &dyn OpeningAccumulator<Fr>) -> OpeningPoint<BIG_ENDIAN, Fr> {
        a.get_committed_polynomial_opening(self.input).0
    }
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (VirtualPoly::NodeOutput(self.namespace), self.input.sumcheck)
    }
    fn log_K(&self) -> usize {
        self.log_k
    }
}
impl RaOneHotEncoding for Lookup {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::NodeOutputRaD(self.namespace, d)
    }
    fn r_cycle_source(&self) -> OpeningId {
        self.input
    }
    fn r_cycle<F: crate::field::JoltField>(&self, a: &dyn OpeningAccumulator<F>) -> Vec<F> {
        a.get_committed_polynomial_opening(self.input).0.r
    }
    fn ra_source(&self) -> OpeningId {
        OpeningId::new(VirtualPoly::NodeOutput(self.namespace), self.input.sumcheck)
    }
    fn log_k(&self) -> usize {
        self.log_k
    }
    fn one_hot_params(&self) -> OneHotParams {
        self.params.clone()
    }
}

impl Lookup {
    pub fn indicator_params(
        &self,
        a: &dyn OpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> shout::RaOneHotParams<Fr> {
        let challenges = shout::ra_onehot_challenges::<Fr, _>(self, t);
        let mut params = shout::ra_onehot_params(self, a, challenges);
        let beta: Fr = t.challenge_scalar();
        let linear = params.1.linear.as_mut().unwrap();
        // Bind the entire chunked address to the input tensor as well as its
        // ReadRaf projection. This also excludes unused leading addresses when
        // the table bit length is not a multiple of the chunk width.
        linear.weights = (0..self.params.instruction_d)
            .map(|d| {
                beta * Fr::from(
                    1u64 << (self.params.log_k_chunk * (self.params.instruction_d - 1 - d)),
                )
            })
            .collect();
        linear.claim.terms.push((self.input, beta));
        params
    }
}

impl NativeLookupStatement {
    fn lookup(&self, max_vars: usize) -> Result<Lookup, ProofVerifyError> {
        if self.table.len() < 2
            || !self.table.len().is_power_of_two()
            || self.log_rows == 0
            || !matches!(self.log_chunk, 1 | 2 | 4 | 8)
            || self
                .log_rows
                .checked_add(self.log_chunk as usize)
                .is_none_or(|n| n > max_vars)
        {
            return Err(invalid("Unsupported native lookup shape"));
        }
        let log_k = self.table.len().ilog2() as usize;
        if log_k.div_ceil(self.log_chunk as usize) * self.log_chunk as usize >= 64 {
            return Err(invalid(
                "Native lookup address exceeds the supported integer width",
            ));
        }
        let params = OneHotParams::from_config_and_log_K(
            &OneHotConfig {
                log_k_chunk: self.log_chunk,
            },
            log_k,
        );
        let lookup = Lookup {
            params,
            log_k,
            input: input_id(),
            output: output_id(),
            namespace: 0,
        };
        let required: BTreeSet<_> = [INPUT, OUTPUT]
            .into_iter()
            .chain((0..lookup.params.instruction_d).map(|d| lookup.committed_poly(d)))
            .collect();
        if !required.iter().eq(self.commitments.keys()) {
            return Err(invalid(
                "Native lookup requires both tensors and every indicator commitment",
            ));
        }
        Ok(lookup)
    }
    fn transcript(&self) -> Blake2bTranscript {
        let mut t = Blake2bTranscript::new(b"Atlas/hidden-table/v1");
        t.append_serializable(self);
        t
    }
    /// Commitments to the entire input and output tensors, under the same
    /// registered Dory setup as this proof. Adjacent proofs must bind these.
    pub fn input_commitment(&self) -> Option<&DoryCommitment> {
        self.commitments.get(&INPUT)
    }
    pub fn output_commitment(&self) -> Option<&DoryCommitment> {
        self.commitments.get(&OUTPUT)
    }
}

impl NativeLookupWitness {
    pub fn commit(
        context: Vec<u8>,
        table: Vec<i32>,
        indices: Vec<usize>,
        log_chunk: u8,
        setup: &DoryProverSetup,
    ) -> Result<(NativeLookupStatement, Self), ProofVerifyError> {
        if indices.len() < 2
            || !indices.len().is_power_of_two()
            || table.len() < 2
            || !table.len().is_power_of_two()
            || indices.iter().any(|i| *i >= table.len())
            || !matches!(log_chunk, 1 | 2 | 4 | 8)
            || indices.len().ilog2() as usize + log_chunk as usize > setup.verifier.max_log_n
        {
            return Err(invalid("Invalid native lookup witness shape or index"));
        }
        let log_rows = indices.len().ilog2() as usize;
        let params = OneHotParams::from_config_and_log_K(
            &OneHotConfig {
                log_k_chunk: log_chunk,
            },
            table.len().ilog2() as usize,
        );
        let mut polynomials = BTreeMap::from([
            (
                INPUT,
                MultilinearPolynomial::from(indices.iter().map(|i| *i as u64).collect::<Vec<_>>()),
            ),
            (
                OUTPUT,
                MultilinearPolynomial::from(indices.iter().map(|i| table[*i]).collect::<Vec<_>>()),
            ),
        ]);
        for d in 0..params.instruction_d {
            polynomials.insert(
                CommittedPoly::NodeOutputRaD(0, d),
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    indices
                        .iter()
                        .map(|i| Some(u16::from(params.lookup_index_chunk(*i as u64, d))))
                        .collect(),
                    params.k_chunk,
                )),
            );
        }
        let mut commitments = BTreeMap::new();
        let mut hints = BTreeMap::new();
        for (id, p) in &polynomials {
            let (c, h) = DoryScheme::commit_zk(p, setup);
            commitments.insert(*id, c);
            hints.insert(*id, h);
        }
        let statement = NativeLookupStatement {
            context,
            table,
            log_rows,
            log_chunk,
            commitments,
        };
        statement.lookup(setup.verifier.max_log_n)?;
        Ok((
            statement,
            Self {
                indices,
                polynomials,
                hints,
            },
        ))
    }
}

impl NativeLookupProof {
    pub fn prove(
        statement: &NativeLookupStatement,
        witness: NativeLookupWitness,
        setup: &DoryProverSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<Self, ProofVerifyError> {
        let lookup = statement.lookup(setup.verifier.max_log_n)?;
        Self::check_generators(&DoryVerifierSetup(setup.verifier.clone()), gens, &lookup)?;
        let NativeLookupWitness {
            indices,
            polynomials,
            hints,
        } = witness;
        if !polynomials.keys().eq(statement.commitments.keys())
            || !hints.keys().eq(statement.commitments.keys())
            || indices.len() != 1 << statement.log_rows
            || indices.iter().any(|i| *i >= statement.table.len())
        {
            return Err(invalid(
                "Native lookup witness does not match the statement",
            ));
        }
        let mut t = statement.transcript();
        let r: Vec<Fr> = t.challenge_vector(statement.log_rows);
        let mut a = ProverOpeningAccumulator::new();
        a.zk_mode = true;
        for id in [input_id(), output_id()] {
            let p = &polynomials[&id.committed_poly().unwrap()];
            a.append_dense(&mut t, id, r.clone(), p.evaluate(&r));
        }
        // Initial claims are constrained by the later opening reduction; they
        // are not outputs of the read sumcheck's transcript commitment block.
        a.take_pending_claims();
        a.take_pending_claim_ids();
        let mut bf = BlindFoldAccumulator::new();
        let mut rng = rand::thread_rng();
        let mut read = shout::read_raf_prover(&lookup, &indices, &statement.table, &a, &mut t);
        let (read, _, _) =
            BatchedSumcheck::prove_zk(vec![read.as_mut()], &mut a, &mut bf, &mut t, gens, &mut rng);
        let [mut ra, mut booleanity] =
            shout::ra_onehot_provers_from_params(lookup.indicator_params(&a, &mut t), &indices);
        let (indicators, _, _) = BatchedSumcheck::prove_zk(
            vec![ra.as_mut(), booleanity.as_mut()],
            &mut a,
            &mut bf,
            &mut t,
            gens,
            &mut rng,
        );
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
            return Err(invalid("Lookup PCS evaluation mismatch"));
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
            read,
            indicators,
            openings,
            pcs,
            blindfold,
        })
    }

    pub fn verify(
        &self,
        statement: &NativeLookupStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        let lookup = statement.lookup(setup.0.max_log_n)?;
        Self::check_generators(setup, gens, &lookup)?;
        let mut t = statement.transcript();
        let r: Vec<Fr> = t.challenge_vector(statement.log_rows);
        let mut a = VerifierOpeningAccumulator::new_zk();
        for id in [input_id(), output_id()] {
            a.append_dense(&mut t, id, r.clone());
        }
        let read = shout::read_raf_verifier(&lookup, statement.table.clone(), &a, &mut t);
        BatchedSumcheck::verify_zk_with_width(
            &self.read,
            vec![read.as_ref()],
            &mut a,
            &mut t,
            gens.message_generators.len(),
        )?;
        let [ra, booleanity] =
            shout::ra_onehot_verifiers_with_params(lookup.indicator_params(&a, &mut t));
        BatchedSumcheck::verify_zk_with_width(
            &self.indicators,
            vec![ra.as_ref(), booleanity.as_ref()],
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
        let commitments: Vec<_> = state
            .polynomials
            .iter()
            .map(|p| statement.commitments[p])
            .collect();
        let joint = DoryScheme::combine_commitments(&commitments, &state.poly_coeffs);
        let eval = self
            .pcs
            .0
            .y_com
            .map(|g| Bn254G1(g.0))
            .ok_or_else(|| invalid("Missing hidden lookup PCS evaluation"))?;
        DoryScheme::verify_zk(&self.pcs, setup, &mut t, &state.r_sumcheck, &eval, &joint)?;
        let native = NativeBlindFold::new(
            a.zk_stages,
            &[relation],
            &coefficients,
            gens.message_generators.len(),
        )?;
        let stages = [&self.read, &self.indicators, &self.openings];
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

    fn check_generators(
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
        lookup: &Lookup,
    ) -> Result<(), ProofVerifyError> {
        super::native_opening::NativeOpeningProof::check_generators(setup, gens)?;
        if gens.message_generators.len() <= lookup.params.instruction_d + 1 {
            return Err(invalid(
                "Native lookup commitment width cannot represent its degree",
            ));
        }
        Ok(())
    }
}

/// A sequence of native table relations with every adjacent hidden tensor
/// bound by a mandatory equality proof. This is a table composition component,
/// not the Qwen generation or zkARc receipt relation.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeLookupChainStatement {
    pub context: Vec<u8>,
    pub stages: Vec<NativeLookupStatement>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeLookupChainProof {
    pub stages: Vec<NativeLookupProof>,
    pub edges: Vec<super::equality::HiddenValueEqualityProof>,
}

impl NativeLookupChainStatement {
    /// Request and stage position are fixed before generating any stage proof.
    pub fn stage_context(context: &[u8], position: usize) -> Vec<u8> {
        let mut bytes = b"Atlas/lookup-chain/stage/v1".to_vec();
        bytes.extend_from_slice(&(context.len() as u64).to_le_bytes());
        bytes.extend_from_slice(context);
        bytes.extend_from_slice(&(position as u64).to_le_bytes());
        bytes
    }

    fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        if self.context.is_empty() || self.stages.len() < 2 {
            return Err(invalid(
                "Native table composition requires a request and adjacent stages",
            ));
        }
        for (i, stage) in self.stages.iter().enumerate() {
            stage.lookup(max_vars)?;
            if stage.log_rows != self.stages[0].log_rows
                || stage.context != Self::stage_context(&self.context, i)
            {
                return Err(invalid(
                    "Native table stage context or tensor shape mismatch",
                ));
            }
        }
        Ok(())
    }

    fn edge(&self, i: usize) -> Result<super::equality::HiddenValueEdge, ProofVerifyError> {
        let mut context = b"Atlas/lookup-chain/edge/v1".to_vec();
        self.serialize_compressed(&mut context)
            .map_err(|_| invalid("Cannot bind native table chain statement"))?;
        context.extend_from_slice(&(i as u64).to_le_bytes());
        Ok(super::equality::HiddenValueEdge {
            context,
            // Both component proofs bind the same ordered field-vector
            // representation. No claim about a different encoding is made.
            encoding: b"bn254-fr-vector-v1".to_vec(),
            shape: vec![1 << self.stages[i].log_rows],
            producer: self.stages[i].commitments[&OUTPUT],
            consumer: self.stages[i + 1].commitments[&INPUT],
        })
    }
}

impl NativeLookupChainProof {
    pub fn prove(
        statement: &NativeLookupChainStatement,
        witnesses: Vec<NativeLookupWitness>,
        setup: &DoryProverSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<Self, ProofVerifyError> {
        statement.validate(setup.verifier.max_log_n)?;
        if witnesses.len() != statement.stages.len() {
            return Err(invalid("Missing native table stage witness"));
        }
        let vp = DoryScheme::setup_verifier(setup);
        let edges = (0..witnesses.len() - 1)
            .map(|i| {
                super::equality::HiddenValueEqualityProof::prove(
                    &statement.edge(i)?,
                    &witnesses[i].hints[&OUTPUT],
                    &witnesses[i + 1].hints[&INPUT],
                    &vp,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stages = statement
            .stages
            .iter()
            .zip(witnesses)
            .map(|(s, w)| NativeLookupProof::prove(s, w, setup, gens))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stages, edges })
    }

    pub fn verify(
        &self,
        statement: &NativeLookupChainStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        statement.validate(setup.0.max_log_n)?;
        if self.stages.len() != statement.stages.len() || self.edges.len() + 1 != self.stages.len()
        {
            return Err(invalid(
                "Every native table stage and adjacent edge is required",
            ));
        }
        for (i, edge) in self.edges.iter().enumerate() {
            edge.verify(&statement.edge(i)?, setup)?;
        }
        for (s, p) in statement.stages.iter().zip(&self.stages) {
            p.verify(s, setup, gens)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::One;

    #[test]
    fn hidden_table_lookup_serialized_proof_binds_every_row_and_stage() {
        let setup = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&setup);
        let gens = DoryScheme::pedersen_generators(&setup, 16);
        // Two chunks, a partially used leading chunk, repeated inputs and
        // signed outputs exercise both address recovery and the linear check.
        let table = vec![-11, 7, 0, 19, 5, -3, 17, 2];
        let (statement, witness) = NativeLookupWitness::commit(
            b"registered-lookup".to_vec(),
            table,
            vec![0, 7, 2, 5, 1, 7, 4, 3],
            2,
            &setup,
        )
        .unwrap();
        let proof = NativeLookupProof::prove(&statement, witness, &setup, &gens).unwrap();
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        let proof = NativeLookupProof::deserialize_compressed(bytes.as_slice()).unwrap();
        proof.verify(&statement, &vp, &gens).unwrap();
        println!("native lookup proof bytes = {}", bytes.len());
        let mut wrong = statement.clone();
        wrong.context.push(1);
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong.table[2] += 1;
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong.log_rows += 1;
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong.commitments.remove(&INPUT);
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong
            .commitments
            .insert(OUTPUT, statement.commitments[&INPUT]);
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.pcs.0.y_com = None;
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.indicators.round_commitments.clear();
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.read.output_claims_commitments[0] += gens.message_generators[0];
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.openings.output_claims_commitments[0] += gens.message_generators[0];
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.blindfold.folded_eval_outputs[0] += Fr::one();
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        let mut wrong = proof.clone();
        wrong.blindfold.folded_eval_outputs.clear();
        assert!(wrong.verify(&statement, &vp, &gens).is_err());
        // A fresh full proof must use new hiding randomness for the same values.
        let (fresh, _) = NativeLookupWitness::commit(
            statement.context.clone(),
            statement.table.clone(),
            vec![0, 7, 2, 5, 1, 7, 4, 3],
            2,
            &setup,
        )
        .unwrap();
        assert_ne!(fresh.commitments[&INPUT], statement.commitments[&INPUT]);
    }

    #[test]
    fn hidden_table_lookup_rejects_false_committed_output_and_address() {
        let setup = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&setup);
        let gens = DoryScheme::pedersen_generators(&setup, 16);
        for bad_input in [false, true] {
            let (mut statement, mut witness) =
                NativeLookupWitness::commit(vec![], vec![9, 5, 2, 7], vec![0, 1, 2, 3], 2, &setup)
                    .unwrap();
            let id = if bad_input { INPUT } else { OUTPUT };
            let values = if bad_input {
                vec![0i64, 1, 2, 0]
            } else {
                vec![9i64, 5, 2, 8]
            };
            let poly = MultilinearPolynomial::from(values);
            let (commitment, hint) = DoryScheme::commit_zk(&poly, &setup);
            statement.commitments.insert(id, commitment);
            witness.polynomials.insert(id, poly);
            witness.hints.insert(id, hint);
            let result = NativeLookupProof::prove(&statement, witness, &setup, &gens);
            if let Ok(proof) = result {
                assert!(proof.verify(&statement, &vp, &gens).is_err());
            }
        }
        assert!(
            NativeLookupWitness::commit(vec![], vec![9, 5, 2, 7], vec![0, 4], 2, &setup).is_err()
        );
        assert!(
            NativeLookupWitness::commit(vec![], vec![9, 5, 2, 7], vec![0, 1], 3, &setup).is_err()
        );
        // Ensure this test also checks the explicit zero-input relation path.
        let zero = crate::subprotocols::booleanity::LinearClaim::constant(Fr::zero());
        let (relation, values) = zero.native_constraint();
        assert!(!relation.terms.is_empty());
        assert_eq!(relation.evaluate::<Fr>(&[], &values), Fr::zero());
        assert_ne!(relation.evaluate::<Fr>(&[], &[Fr::one()]), Fr::zero());
    }
    #[test]
    fn hidden_table_chain_requires_the_actual_producer_and_consumer_tensors() {
        let pp = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let request = b"registered-request".to_vec();
        let ctx = |i| NativeLookupChainStatement::stage_context(&request, i);
        let (s0, w0) =
            NativeLookupWitness::commit(ctx(0), vec![3, 0, 2, 1], vec![0, 1, 3, 2], 2, &pp)
                .unwrap();
        let (s1, w1) =
            NativeLookupWitness::commit(ctx(1), vec![9, 7, -2, 5], vec![3, 0, 1, 2], 2, &pp)
                .unwrap();
        let statement = NativeLookupChainStatement {
            context: request.clone(),
            stages: vec![s0, s1],
        };
        let proof = NativeLookupChainProof::prove(&statement, vec![w0, w1], &pp, &gens).unwrap();
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        let decoded = NativeLookupChainProof::deserialize_compressed(bytes.as_slice()).unwrap();
        decoded.verify(&statement, &vp, &gens).unwrap();
        let mut missing = decoded.clone();
        missing.edges.clear();
        assert!(missing.verify(&statement, &vp, &gens).is_err());
        let mut missing = decoded.clone();
        missing.stages.pop();
        assert!(missing.verify(&statement, &vp, &gens).is_err());
        let mut swapped = decoded.clone();
        swapped.stages.swap(0, 1);
        assert!(swapped.verify(&statement, &vp, &gens).is_err());
        let mut wrong_request = statement.clone();
        wrong_request.context.push(1);
        assert!(decoded.verify(&wrong_request, &vp, &gens).is_err());
        // This consumer proof is independently valid, but consumes a different
        // hidden vector. A host label or standalone proof would not catch it.
        let (bad_statement, bad_witness) =
            NativeLookupWitness::commit(ctx(1), vec![9, 7, -2, 5], vec![3, 0, 1, 3], 2, &pp)
                .unwrap();
        let bad_proof = NativeLookupProof::prove(&bad_statement, bad_witness, &pp, &gens).unwrap();
        bad_proof.verify(&bad_statement, &vp, &gens).unwrap();
        let mut mixed_statement = statement.clone();
        mixed_statement.stages[1] = bad_statement;
        let mut mixed = decoded;
        mixed.stages[1] = bad_proof;
        assert!(mixed.verify(&mixed_statement, &vp, &gens).is_err());
        println!("native lookup chain proof bytes = {}", bytes.len());
    }
}
