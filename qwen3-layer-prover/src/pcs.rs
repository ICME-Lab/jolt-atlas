use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    layer::{LayerShape, LayerTensorIds, LayerWeights, LayerWitness},
    ops::round::ROUND_LUT_LEN,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerCommittedPoly {
    pub name: String,
}

impl LayerCommittedPoly {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone)]
pub struct LayerPolynomialEntry<F: JoltField> {
    pub id: LayerCommittedPoly,
    pub poly: MultilinearPolynomial<F>,
}

#[derive(Debug, Clone, Default)]
pub struct LayerPolynomialMap<F: JoltField> {
    pub entries: Vec<LayerPolynomialEntry<F>>,
}

#[derive(Debug, Clone)]
pub struct LayerCommitmentEntry<C> {
    pub id: LayerCommittedPoly,
    pub commitment: C,
}

#[derive(Debug, Clone, Default)]
pub struct LayerCommitments<C> {
    pub entries: Vec<LayerCommitmentEntry<C>>,
}

#[derive(Debug, Clone)]
pub struct LayerPcsCommitments<PCS: CommitmentScheme> {
    pub commitments: LayerCommitments<PCS::Commitment>,
    pub hints: Vec<PCS::OpeningProofHint>,
}

#[derive(Debug, Clone)]
pub struct LayerOpeningReductionProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>
{
    // Opening requests are exactly the Claim values returned by prove_xxx.
    // Individual op provers stay pure: they do not see a PCS accumulator and
    // they do not mutate layer-level opening state. The outer layer wrapper is
    // responsible for reducing these requests and producing a PCS proof.
    pub opening_claims: Vec<crate::Claim<F>>,
    pub reduced_opening: jolt_atlas_core::onnx_proof::ReducedOpeningProof<F, T, PCS>,
}

impl<F: JoltField> LayerPolynomialMap<F> {
    pub fn from_layer(
        hidden_out: &[i32],
        witness: &LayerWitness,
        _weights: &LayerWeights,
        shape: &LayerShape,
        tensors: &LayerTensorIds,
    ) -> Self {
        let mut out = Self::default();
        out.push_i32("hidden_out", hidden_out);

        // Commit witness/advice tensors. We intentionally do not expose any
        // accumulator here; the later layer-level PCS wrapper will decide which
        // of these committed polynomials are opened by the op proofs.
        out.push_i32(tensors.hidden_in_a.clone(), &witness.hidden_in);
        out.push_i32(tensors.hidden_in_b.clone(), &witness.hidden_in);
        out.push_i64(
            "rms_norm_atten_sum_x2",
            &witness.rms_norm_atten_sum_x2,
        );
        out.push_i64(
            "rms_norm_atten_norm_acc",
            &witness.rms_norm_atten_norm_acc,
        );
        out.push_i32(
            "rms_norm_atten_norm",
            &witness.rms_norm_atten_norm,
        );
        out.push_u8_bits(
            "rms_norm_atten_norm_frac_bits",
            &witness.rms_norm_atten_norm_frac_bits,
        );
        out.push_i64(
            "rms_norm_atten_acc",
            &witness.rms_norm_atten_acc,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_atten_acc),
            &witness.rms_norm_atten_acc,
            &[witness.rms_norm_atten_a.len()],
        );
        out.push_i32(
            "rms_norm_atten_a",
            &witness.rms_norm_atten_a,
        );
        out.push_i32(
            "rms_norm_atten_b",
            &witness.rms_norm_atten_b,
        );
        out.push_i32(
            "rms_norm_atten_c",
            &witness.rms_norm_atten_c,
        );
        out.push_u8_bits(
            "rms_norm_atten_frac_bits",
            &witness.rms_norm_atten_frac_bits,
        );

        out.push_i64("context_acc", &witness.context_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.context_acc),
            &witness.context_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
        );
        out.push_i32("context", &witness.context);
        out.push_u8_bits(
            "context_frac_bits",
            &witness.context_frac_bits,
        );
        out.push_i32("o_proj", &witness.o_proj);
        out.push_i64("o_proj_acc", &witness.o_proj_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.o_proj_acc),
            &witness.o_proj_acc,
            &[shape.seq, shape.hidden],
        );
        out.push_u8_bits(
            "o_proj_frac_bits",
            &witness.o_proj_frac_bits,
        );

        out.push_i32("softmax", &witness.softmax);
        out.push_i64("softmax_acc", &witness.softmax_acc);
        let softmax_floor_as_i64 = witness
            .softmax_floor
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let softmax_acc_shifted = witness
            .softmax_acc
            .iter()
            .map(|&value| value - (1_i64 << 7))
            .collect::<Vec<_>>();
        out.push_round_ra(
            "softmax_floor_round_ra",
            &softmax_floor_as_i64,
            &[shape.q_heads, shape.seq, shape.seq],
        );
        out.push_round_ra(
            "softmax_acc_shifted_round_ra",
            &softmax_acc_shifted,
            &[shape.q_heads, shape.seq, shape.seq],
        );
        out.push_i32(
            "softmax_floor",
            &witness.softmax_floor,
        );
        out.push_u8_bits(
            "softmax_floor_frac_bits",
            &witness.softmax_floor_frac_bits,
        );
        out.push_u8_bits(
            "softmax_frac_bits",
            &witness.softmax_frac_bits,
        );
        out.push_usize_as_u64(
            "softmax_max_index",
            &witness.softmax_max_index,
        );
        out.push_lookup_ra(
            "softmax_ra",
            &witness.softmax_ra,
            &[shape.q_heads, shape.seq, shape.seq],
            softmax_entries(witness.softmax_min_diff, witness.softmax_max_diff),
            padded_softmax_lut_len(softmax_entries(
                witness.softmax_min_diff,
                witness.softmax_max_diff,
            )),
            softmax_entries(witness.softmax_min_diff, witness.softmax_max_diff),
        );
        out.push_softmax_input_remainder_ra(
            "softmax_input_remainder_ra",
            &witness.qk_score,
            &witness.softmax_max,
            shape.q_heads,
            shape.seq,
        );
        out.push_i64(
            "softmax_exp_acc",
            &witness.softmax_exp_acc,
        );
        out.push_u8_bits(
            "softmax_exp_frac_bits",
            &witness.softmax_exp_frac_bits,
        );
        out.push_i32("softmax_max", &witness.softmax_max);
        out.push_i32("softmax_exp", &witness.softmax_exp);
        out.push_i32("softmax_sum", &witness.softmax_sum);

        out.push_i32("qk_score", &witness.qk_score);
        out.push_i64("qk_score_acc", &witness.qk_score_acc);
        out.push_i32("qk_score_dot", &witness.qk_score_dot);
        out.push_u8_bits(
            "qk_score_dot_frac_bits",
            &witness.qk_score_dot_frac_bits,
        );
        out.push_i64(
            "qk_score_scale_acc",
            &witness.qk_score_scale_acc,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.qk_score_scale_acc),
            &witness.qk_score_scale_acc,
            &[shape.q_heads, shape.seq, shape.seq],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.qk_score_acc),
            &witness.qk_score_acc,
            &[shape.q_heads, shape.seq, shape.seq],
        );
        out.push_u8_bits(
            "qk_score_frac_bits",
            &witness.qk_score_frac_bits,
        );

        out.push_i64("q_rope_acc", &witness.q_rope_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.q_rope_acc),
            &witness.q_rope_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
        );
        out.push_i32("q_rope", &witness.q_rope);
        out.push_u8_bits(
            "q_rope_frac_bits",
            &witness.q_rope_frac_bits,
        );
        out.push_i64("k_rope_acc", &witness.k_rope_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_rope_acc),
            &witness.k_rope_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
        );
        out.push_i32("k_rope", &witness.k_rope);
        out.push_u8_bits(
            "k_rope_frac_bits",
            &witness.k_rope_frac_bits,
        );

        out.push_i32("q_proj", &witness.q_proj);
        out.push_i32("k_proj", &witness.k_proj);
        out.push_i32("v_proj", &witness.v_proj);
        out.push_i64("q_proj_acc", &witness.q_proj_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.q_proj_acc),
            &witness.q_proj_acc,
            &[shape.seq, shape.attention_width()],
        );
        out.push_u8_bits(
            "q_proj_frac_bits",
            &witness.q_proj_frac_bits,
        );
        out.push_i64("k_proj_acc", &witness.k_proj_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_proj_acc),
            &witness.k_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
        );
        out.push_u8_bits(
            "k_proj_frac_bits",
            &witness.k_proj_frac_bits,
        );
        out.push_i64("v_proj_acc", &witness.v_proj_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.v_proj_acc),
            &witness.v_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
        );
        out.push_u8_bits(
            "v_proj_frac_bits",
            &witness.v_proj_frac_bits,
        );

        out.push_i64(
            "q_norm_sum_x2",
            &witness.q_norm_sum_x2,
        );
        out.push_i64(
            "k_norm_sum_x2",
            &witness.k_norm_sum_x2,
        );
        out.push_i64(
            "q_norm_norm_acc",
            &witness.q_norm_norm_acc,
        );
        out.push_i64(
            "k_norm_norm_acc",
            &witness.k_norm_norm_acc,
        );
        out.push_i32("q_norm_norm", &witness.q_norm_norm);
        out.push_i32("k_norm_norm", &witness.k_norm_norm);
        out.push_u8_bits(
            "q_norm_norm_frac_bits",
            &witness.q_norm_norm_frac_bits,
        );
        out.push_u8_bits(
            "k_norm_norm_frac_bits",
            &witness.k_norm_norm_frac_bits,
        );
        out.push_i64("q_norm_acc", &witness.q_norm_acc);
        out.push_i64("k_norm_acc", &witness.k_norm_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.q_norm_acc),
            &witness.q_norm_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_norm_acc),
            &witness.k_norm_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
        );
        out.push_i32("q_norm", &witness.q_norm);
        out.push_i32("k_norm", &witness.k_norm);
        out.push_u8_bits(
            "q_norm_frac_bits",
            &witness.q_norm_frac_bits,
        );
        out.push_u8_bits(
            "k_norm_frac_bits",
            &witness.k_norm_frac_bits,
        );

        out.push_i32(
            "residual_add_attn_a",
            &witness.residual_add_attn_a,
        );
        out.push_i32(
            "residual_add_attn_b",
            &witness.residual_add_attn_b,
        );
        out.push_i64(
            "rms_norm_mlp_sum_x2",
            &witness.rms_norm_mlp_sum_x2,
        );
        out.push_i64(
            "rms_norm_mlp_norm_acc",
            &witness.rms_norm_mlp_norm_acc,
        );
        out.push_i32(
            "rms_norm_mlp_norm",
            &witness.rms_norm_mlp_norm,
        );
        out.push_u8_bits(
            "rms_norm_mlp_norm_frac_bits",
            &witness.rms_norm_mlp_norm_frac_bits,
        );
        out.push_i64(
            "rms_norm_mlp_acc",
            &witness.rms_norm_mlp_acc,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_mlp_acc),
            &witness.rms_norm_mlp_acc,
            &[shape.seq, shape.hidden],
        );
        out.push_i32(
            "rms_norm_mlp_a",
            &witness.rms_norm_mlp_a,
        );
        out.push_i32(
            "rms_norm_mlp_b",
            &witness.rms_norm_mlp_b,
        );
        out.push_u8_bits(
            "rms_norm_mlp_frac_bits",
            &witness.rms_norm_mlp_frac_bits,
        );

        out.push_i64(
            "gate_proj_acc",
            &witness.gate_proj_acc,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.gate_proj_acc),
            &witness.gate_proj_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_i32("gate_proj", &witness.gate_proj);
        out.push_u8_bits(
            "gate_proj_frac_bits",
            &witness.gate_proj_frac_bits,
        );
        out.push_i64("up_proj_acc", &witness.up_proj_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.up_proj_acc),
            &witness.up_proj_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_i32("up_proj", &witness.up_proj);
        out.push_u8_bits(
            "up_proj_frac_bits",
            &witness.up_proj_frac_bits,
        );
        out.push_i64("silu_acc", &witness.silu_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.gate_proj),
            &witness.gate_proj.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
            &[shape.seq, shape.intermediate],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.silu_acc),
            &witness.silu_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_i32("silu", &witness.silu);
        out.push_u8_bits(
            "silu_frac_bits",
            &witness.silu_frac_bits,
        );
        out.push_lookup_ra(
            "silu_ra",
            &witness.silu_ra,
            &[shape.seq, shape.intermediate],
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            padded_silu_lut_len(silu_entries(witness.silu_min_n, witness.silu_max_n)),
            silu_entries(witness.silu_min_n, witness.silu_max_n),
        );
        out.push_u8_bits(
            "silu_out_frac_bits",
            &witness.silu_out_frac_bits,
        );
        out.push_i64("silu_up_acc", &witness.silu_up_acc);
        out.push_round_ra(
            format!("{}_round_ra", tensors.silu_up_acc),
            &witness.silu_up_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_i32("silu_up", &witness.silu_up);
        out.push_u8_bits(
            "silu_up_frac_bits",
            &witness.silu_up_frac_bits,
        );
        out.push_i64(
            "down_proj_acc",
            &witness.down_proj_acc,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.down_proj_acc),
            &witness.down_proj_acc,
            &[shape.seq, shape.hidden],
        );
        out.push_i32("down_proj", &witness.down_proj);
        out.push_u8_bits(
            "down_proj_frac_bits",
            &witness.down_proj_frac_bits,
        );
        out
    }

    pub fn has_tensor(&self, tensor: &str) -> bool {
        self.entries.iter().any(|entry| entry.id.name == tensor)
    }

    pub fn committed_poly_for_tensor(&self, tensor: &str) -> Option<CommittedPoly> {
        self.entries
            .iter()
            .position(|entry| entry.id.name == tensor)
            .map(CommittedPoly::QwenLayerTensor)
    }

    pub fn core_poly_map_for_claims(
        &self,
        claims: &[crate::Claim<F>],
    ) -> Result<BTreeMap<CommittedPoly, MultilinearPolynomial<F>>, crate::ProverError> {
        let mut out = BTreeMap::new();
        for claim in claims {
            let Some((idx, entry)) = self
                .entries
                .iter()
                .enumerate()
                .find(|(_, entry)| entry.id.name == claim.tensor.0)
            else {
                return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                    claim.tensor.0.clone(),
                ]));
            };
            let expected = claim.domain_shape.numel();
            let actual = entry.poly.len();
            if actual != expected {
                return Err(crate::ProverError::CommittedPolynomialDomainMismatch {
                    tensor: claim.tensor.0.clone(),
                    domain_shape: claim.domain_shape.0.clone(),
                    expected,
                    actual,
                });
            }
            out.insert(CommittedPoly::QwenLayerTensor(idx), entry.poly.clone());
        }
        Ok(out)
    }

    pub fn missing_opening_claims(&self, claims: &[crate::Claim<F>]) -> Vec<String> {
        let committed = self
            .entries
            .iter()
            .map(|entry| entry.id.name.as_str())
            .collect::<BTreeSet<_>>();
        claims
            .iter()
            .filter_map(|claim| {
                let name = claim.tensor.0.as_str();
                (!committed.contains(name)).then(|| name.to_string())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    #[cfg(test)]
    pub fn opening_value_mismatches(&self, claims: &[crate::Claim<F>]) -> Vec<String> {
        let mut mismatches = Vec::new();
        for claim in claims {
            let Some(entry) = self
                .entries
                .iter()
                .find(|entry| entry.id.name == claim.tensor.0)
            else {
                mismatches.push(format!("{}: missing", claim.tensor.0));
                continue;
            };
            if entry.poly.len() != claim.domain_shape.numel() {
                mismatches.push(format!(
                    "{}: len {} != {}",
                    claim.tensor.0,
                    entry.poly.len(),
                    claim.domain_shape.numel()
                ));
                continue;
            }
            let value =
                joltworks::poly::multilinear_polynomial::PolynomialEvaluation::evaluate(
                    &entry.poly,
                    &claim.point,
                );
            if value != claim.value {
                mismatches.push(format!("{}: value mismatch", claim.tensor.0));
            }
        }
        mismatches
    }

    fn push_i32(&mut self, name: impl Into<String>, values: &[i32]) {
        self.entries.push(LayerPolynomialEntry {
            id: LayerCommittedPoly::new(name),
            poly: MultilinearPolynomial::from(pad_power_of_two(values)),
        });
    }

    fn push_i64(&mut self, name: impl Into<String>, values: &[i64]) {
        self.entries.push(LayerPolynomialEntry {
            id: LayerCommittedPoly::new(name),
            poly: MultilinearPolynomial::from(pad_power_of_two(values)),
        });
    }

    fn push_u8(&mut self, name: impl Into<String>, values: &[u8]) {
        self.entries.push(LayerPolynomialEntry {
            id: LayerCommittedPoly::new(name),
            poly: MultilinearPolynomial::from(pad_power_of_two(values)),
        });
    }

    fn push_round_ra(
        &mut self,
        name: impl Into<String>,
        input: &[i64],
        logical_shape: &[usize],
    ) {
        let logical_len = logical_shape.iter().product::<usize>();
        debug_assert_eq!(logical_len, input.len());
        let padded_len = logical_shape
            .iter()
            .map(|dim| dim.next_power_of_two())
            .product::<usize>();
        let mut values = vec![0_u8; ROUND_LUT_LEN * padded_len];
        for row in 0..padded_len {
            // Shout's RA virtual polynomial is indexed as ra(address, cycle):
            // the lookup-table address bits are the first coordinates in the
            // opening point, and the tensor/cycle bits come after them.
            values[row] = 1;
        }
        let strides = row_major_strides(logical_shape);
        let padded_dims = logical_shape
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<_>>();
        let padded_strides = row_major_strides(&padded_dims);
        for (flat, &value) in input.iter().enumerate() {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % logical_shape[dim];
                padded_flat += coord * padded_stride;
            }
            let remainder = value.rem_euclid(ROUND_LUT_LEN as i64) as usize;
            values[padded_flat] = 0;
            values[remainder * padded_len + padded_flat] = 1;
        }
        self.push_u8(name, &values);
    }

    fn push_softmax_input_remainder_ra(
        &mut self,
        name: impl Into<String>,
        input: &[i32],
        row_max: &[i32],
        heads: usize,
        seq: usize,
    ) {
        let rows = heads * seq;
        debug_assert_eq!(input.len(), rows * seq);
        debug_assert_eq!(row_max.len(), rows);
        let padded_len = rows * seq;
        let mut values = vec![0_u8; ROUND_LUT_LEN * padded_len];
        for row in 0..rows {
            let query = row % seq;
            for key in 0..seq {
                let idx = row * seq + key;
                let diff = if key <= query {
                    i64::from(input[idx]) - i64::from(row_max[row])
                } else {
                    0
                };
                let rem = diff.rem_euclid(ROUND_LUT_LEN as i64) as usize;
                values[rem * padded_len + idx] = 1;
            }
        }
        self.push_u8(name, &values);
    }

    fn push_lookup_ra(
        &mut self,
        name: impl Into<String>,
        logical_onehot: &[u8],
        logical_shape: &[usize],
        logical_entries: usize,
        padded_entries: usize,
        padding_index: usize,
    ) {
        let logical_len = logical_shape.iter().product::<usize>();
        debug_assert_eq!(logical_onehot.len(), logical_len * logical_entries);
        debug_assert!(logical_entries <= padded_entries);
        debug_assert!(padding_index < padded_entries);

        let padded_dims = logical_shape
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<_>>();
        let padded_len = padded_dims.iter().product::<usize>();
        let mut values = vec![0_u8; padded_entries * padded_len];
        for row in 0..padded_len {
            values[padding_index * padded_len + row] = 1;
        }

        let strides = row_major_strides(logical_shape);
        let padded_strides = row_major_strides(&padded_dims);
        for flat in 0..logical_len {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % logical_shape[dim];
                padded_flat += coord * padded_stride;
            }
            let src = flat * logical_entries;
            let entry = logical_onehot[src..src + logical_entries]
                .iter()
                .position(|&value| value != 0)
                .unwrap_or(padding_index);
            debug_assert!(entry < padded_entries);
            values[padding_index * padded_len + padded_flat] = 0;
            values[entry * padded_len + padded_flat] = 1;
        }

        self.entries.push(LayerPolynomialEntry {
            id: LayerCommittedPoly::new(name),
            poly: MultilinearPolynomial::from(values),
        });
    }

    fn push_usize_as_u64(&mut self, name: impl Into<String>, values: &[usize]) {
        let values = values.iter().map(|&v| v as u64).collect::<Vec<_>>();
        self.entries.push(LayerPolynomialEntry {
            id: LayerCommittedPoly::new(name),
            poly: MultilinearPolynomial::from(pad_power_of_two(&values)),
        });
    }

    fn push_u8_bits<const N: usize>(&mut self, name: impl Into<String>, values: &[Vec<u8>; N]) {
        let name = name.into();
        for (idx, bit_values) in values.iter().enumerate() {
            self.push_u8(format!("{name}.{idx}"), bit_values);
        }
    }
}

pub fn prove_layer_openings_core_style<F, T, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    opening_claims: Vec<crate::Claim<F>>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<LayerOpeningReductionProof<F, T, PCS>, crate::ProverError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let poly_map = polynomials.core_poly_map_for_claims(&opening_claims)?;
    let mut accumulator = ProverOpeningAccumulator::new();
    append_claims_to_prover_accumulator(polynomials, &opening_claims, &mut accumulator, transcript)?;
    let reduced_opening = jolt_atlas_core::opening_reduction::prove_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &poly_map,
        setup,
        transcript,
    )
    .ok_or(crate::ProverError::MissingOpening)?;
    Ok(LayerOpeningReductionProof {
        opening_claims,
        reduced_opening,
    })
}

pub fn verify_layer_openings_core_style<F, T, PCS>(
    commitments: &LayerCommitments<PCS::Commitment>,
    proof: &LayerOpeningReductionProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let mut accumulator = VerifierOpeningAccumulator::new();
    append_claims_to_verifier_accumulator(
        commitments,
        &proof.opening_claims,
        &mut accumulator,
        transcript,
    )?;
    let core_commitments = commitments_as_core_vec(commitments, &proof.opening_claims)?;
    jolt_atlas_core::opening_reduction::verify_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &core_commitments,
        &proof.reduced_opening,
        setup,
        transcript,
    )
}

fn append_claims_to_prover_accumulator<F, T>(
    polynomials: &LayerPolynomialMap<F>,
    claims: &[crate::Claim<F>],
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), crate::ProverError>
where
    F: JoltField,
    T: Transcript,
{
    let mut seen = BTreeSet::new();
    for (idx, claim) in claims.iter().enumerate() {
        let Some(poly) = polynomials.committed_poly_for_tensor(&claim.tensor.0) else {
            return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                claim.tensor.0.clone(),
            ]));
        };
        let entry = &polynomials.entries[match poly {
            CommittedPoly::QwenLayerTensor(idx) => idx,
            _ => unreachable!("qwen layer helper only produces QwenLayerTensor ids"),
        }];
        let expected = claim.domain_shape.numel();
        let actual = entry.poly.len();
        if actual != expected {
            return Err(crate::ProverError::CommittedPolynomialDomainMismatch {
                tensor: claim.tensor.0.clone(),
                domain_shape: claim.domain_shape.0.clone(),
                expected,
                actual,
            });
        }
        if !seen.insert(poly) {
            return Err(crate::ProverError::MissingCommittedPolynomials(vec![format!(
                "duplicate opening for {:?}",
                claim.tensor.0
            )]));
        }
        accumulator.append_dense(
            transcript,
            OpeningId::new(poly, SumcheckId::NodeExecution(idx)),
            claim.point.clone(),
            claim.value,
        );
    }
    Ok(())
}

fn append_claims_to_verifier_accumulator<F, T>(
    commitments: &LayerCommitments<impl Clone>,
    claims: &[crate::Claim<F>],
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let mut seen = BTreeSet::new();
    for (idx, claim) in claims.iter().enumerate() {
        let Some(poly_idx) = commitments
            .entries
            .iter()
            .position(|entry| entry.id.name == claim.tensor.0)
        else {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "missing commitment for tensor {}",
                claim.tensor.0
            )));
        };
        let poly = CommittedPoly::QwenLayerTensor(poly_idx);
        if !seen.insert(poly) {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "duplicate opening for tensor {}",
                claim.tensor.0
            )));
        }
        let opening_id = OpeningId::new(poly, SumcheckId::NodeExecution(idx));
        accumulator
            .openings
            .insert(opening_id, (OpeningPoint::default(), claim.value));
        accumulator.append_dense(transcript, opening_id, claim.point.clone());
    }
    Ok(())
}

fn commitments_as_core_vec<F: JoltField, C: Clone>(
    commitments: &LayerCommitments<C>,
    claims: &[crate::Claim<F>],
) -> Result<Vec<C>, ProofVerifyError> {
    let mut out = BTreeMap::new();
    for claim in claims {
        let Some(poly_idx) = commitments
            .entries
            .iter()
            .position(|entry| entry.id.name == claim.tensor.0)
        else {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "missing commitment for tensor {}",
                claim.tensor.0
            )));
        };
        out.insert(
            CommittedPoly::QwenLayerTensor(poly_idx),
            commitments.entries[poly_idx].commitment.clone(),
        );
    }
    Ok(out.into_values().collect())
}

fn pad_power_of_two<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let len = values.len().next_power_of_two();
    let mut out = vec![T::default(); len];
    out[..values.len()].copy_from_slice(values);
    out
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for idx in (0..dims.len()).rev() {
        if idx + 1 < dims.len() {
            strides[idx] = strides[idx + 1] * dims[idx + 1];
        }
    }
    strides
}

fn softmax_entries(min_diff: i64, max_diff: i64) -> usize {
    debug_assert!(max_diff >= min_diff);
    (max_diff - min_diff + 1) as usize
}

fn padded_softmax_lut_len(entries: usize) -> usize {
    (entries + 1).next_power_of_two().max(16)
}

fn silu_entries(min_n: i64, max_n: i64) -> usize {
    debug_assert!(max_n >= min_n);
    (max_n - min_n + 1) as usize
}

fn padded_silu_lut_len(entries: usize) -> usize {
    (entries + 1).next_power_of_two().max(16)
}

pub fn commit_layer_polynomials<F, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    setup: &PCS::ProverSetup,
) -> LayerPcsCommitments<PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut commitments = Vec::with_capacity(polynomials.entries.len());
    let mut hints = Vec::with_capacity(polynomials.entries.len());
    for entry in &polynomials.entries {
        let (commitment, hint) = PCS::commit(&entry.poly, setup);
        commitments.push(LayerCommitmentEntry {
            id: entry.id.clone(),
            commitment,
        });
        hints.push(hint);
    }
    LayerPcsCommitments {
        commitments: LayerCommitments {
            entries: commitments,
        },
        hints,
    }
}

pub fn absorb_layer_commitments<F, T, C>(
    transcript: &mut T,
    layer: usize,
    shape: &LayerShape,
    hidden_out_a: &crate::Claim<F>,
    hidden_out_b: &crate::Claim<F>,
    commitments: &LayerCommitments<C>,
) where
    F: JoltField,
    T: Transcript,
    C: AppendToTranscript,
{
    transcript.append_message(b"qwen3-layer-pcs-v1");
    transcript.append_u64(layer as u64);
    for &dim in &[
        shape.seq,
        shape.hidden,
        shape.intermediate,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
    ] {
        transcript.append_u64(dim as u64);
    }
    absorb_claim(transcript, hidden_out_a);
    absorb_claim(transcript, hidden_out_b);
    transcript.append_u64(commitments.entries.len() as u64);
    for entry in &commitments.entries {
        transcript.append_bytes(entry.id.name.as_bytes());
        entry.commitment.append_to_transcript(transcript);
    }
}

fn absorb_claim<F, T>(transcript: &mut T, claim: &crate::Claim<F>)
where
    F: JoltField,
    T: Transcript,
{
    transcript.append_bytes(claim.tensor.0.as_bytes());
    for &dim in claim.logical_shape.dims() {
        transcript.append_u64(dim as u64);
    }
    transcript.append_scalars(&claim.point);
    transcript.append_scalar(&claim.value);
}
