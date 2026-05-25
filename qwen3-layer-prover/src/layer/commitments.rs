use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial,
    },
    transcripts::{AppendToTranscript, Transcript},
};
use std::collections::{BTreeMap, BTreeSet};

use crate::ops::round::ROUND_LUT_LEN;

use super::{
    tensors::LayerTensorIds,
    types::{LayerShape, LayerWeights},
    witness::LayerWitness,
};

// Commitments for one layer. This file answers what is committed and how those
// commitments are bound into the transcript. The PCS opening proof is
// in `openings.rs`.

#[derive(Debug, Clone)]
pub struct LayerPolynomialEntry<F: JoltField> {
    pub name: String,
    pub poly: MultilinearPolynomial<F>,
}

#[derive(Debug, Clone, Default)]
pub struct LayerPolynomialMap<F: JoltField> {
    pub entries: Vec<LayerPolynomialEntry<F>>,
}

#[derive(Debug, Clone)]
pub struct LayerCommitmentEntry<C> {
    pub name: String,
    pub commitment: C,
}

#[derive(Debug, Clone, Default)]
pub struct LayerCommitments<C> {
    pub entries: Vec<LayerCommitmentEntry<C>>,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerCommitmentsAndPolynomials<F, C>
where
    F: JoltField,
{
    pub polynomials: LayerPolynomialMap<F>,
    pub commitments: LayerCommitments<C>,
}

#[derive(Debug, Clone)]
pub struct HiddenStateCommitments<C> {
    pub hidden_in: C,
    pub hidden_out: C,
}

impl<C: Clone> LayerCommitments<C> {
    pub fn with_hidden_state_commitments(
        &self,
        hidden_state_commitments: HiddenStateCommitments<C>,
    ) -> Self {
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                let commitment = match entry.name.as_str() {
                    "hidden_in" => hidden_state_commitments.hidden_in.clone(),
                    "hidden_out" => hidden_state_commitments.hidden_out.clone(),
                    _ => entry.commitment.clone(),
                };
                LayerCommitmentEntry {
                    name: entry.name.clone(),
                    commitment,
                }
            })
            .collect();
        Self { entries }
    }
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

        // Commit only the claims that leave the layer IOP unresolved.
        // Regular witness tensors and accumulators are consumed inside op
        // sumchecks; committing them here is both unnecessary and too large for
        // real traces.
        out.push_i32("hidden_out", hidden_out);
        out.push_i32(tensors.hidden_in_a.clone(), &witness.hidden_in);

        out.push_round_ra(
            format!("{}_round_ra", tensors.context_acc),
            &witness.context_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.o_proj_acc),
            &witness.o_proj_acc,
            &[shape.seq, shape.hidden],
        );

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

        out.push_round_ra(
            format!("{}_round_ra", tensors.q_rope_acc),
            &witness.q_rope_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_rope_acc),
            &witness.k_rope_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.q_proj_acc),
            &witness.q_proj_acc,
            &[shape.seq, shape.attention_width()],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_proj_acc),
            &witness.k_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.v_proj_acc),
            &witness.v_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
        );

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

        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_atten_acc),
            &witness.rms_norm_atten_acc,
            &[witness.rms_norm_atten_a.len()],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_mlp_acc),
            &witness.rms_norm_mlp_acc,
            &[shape.seq, shape.hidden],
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.gate_proj_acc),
            &witness.gate_proj_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.up_proj_acc),
            &witness.up_proj_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.gate_proj),
            &witness
                .gate_proj
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>(),
            &[shape.seq, shape.intermediate],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.silu_acc),
            &witness.silu_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_lookup_ra(
            "silu_ra",
            &witness.silu_ra,
            &[shape.seq, shape.intermediate],
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            padded_silu_lut_len(silu_entries(witness.silu_min_n, witness.silu_max_n)),
            silu_entries(witness.silu_min_n, witness.silu_max_n),
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.silu_up_acc),
            &witness.silu_up_acc,
            &[shape.seq, shape.intermediate],
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.down_proj_acc),
            &witness.down_proj_acc,
            &[shape.seq, shape.hidden],
        );
        out
    }

    pub fn committed_poly_for_tensor(&self, tensor: &str) -> Option<CommittedPoly> {
        self.entries
            .iter()
            .position(|entry| entry.name == tensor)
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
                .find(|(_, entry)| entry.name == claim.tensor.0)
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
            .map(|entry| entry.name.as_str())
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
                .find(|entry| entry.name == claim.tensor.0)
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
            let value = joltworks::poly::multilinear_polynomial::PolynomialEvaluation::evaluate(
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
            name: name.into(),
            poly: MultilinearPolynomial::from(pad_power_of_two(values)),
        });
    }

    fn push_round_ra(&mut self, name: impl Into<String>, input: &[i64], logical_shape: &[usize]) {
        let logical_len = logical_shape.iter().product::<usize>();
        debug_assert_eq!(logical_len, input.len());
        let padded_dims = power_of_two_dims(logical_shape);
        let padded_len = padded_dims.iter().product::<usize>();
        let mut indices = vec![Some(0_u16); padded_len];
        let strides = row_major_strides(logical_shape);
        let padded_strides = row_major_strides(&padded_dims);
        for (flat, &value) in input.iter().enumerate() {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % logical_shape[dim];
                padded_flat += coord * padded_stride;
            }
            let remainder = value.rem_euclid(ROUND_LUT_LEN as i64) as usize;
            indices[padded_flat] = Some(remainder as u16);
        }
        self.push_one_hot(name, indices, ROUND_LUT_LEN);
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
        let logical_shape = [heads, seq, seq];
        let padded_dims = power_of_two_dims(&logical_shape);
        let padded_len = padded_dims.iter().product::<usize>();
        let strides = row_major_strides(&logical_shape);
        let padded_strides = row_major_strides(&padded_dims);
        let mut indices = vec![None; padded_len];
        for (row, &max) in row_max.iter().enumerate().take(rows) {
            let query = row % seq;
            for key in 0..seq {
                let idx = row * seq + key;
                let flat = row * seq + key;
                let mut padded_flat = 0;
                for (dim, (&stride, &padded_stride)) in
                    strides.iter().zip(&padded_strides).enumerate()
                {
                    let coord = (flat / stride) % logical_shape[dim];
                    padded_flat += coord * padded_stride;
                }
                let diff = if key <= query {
                    i64::from(input[idx]) - i64::from(max)
                } else {
                    0
                };
                let rem = diff.rem_euclid(ROUND_LUT_LEN as i64) as usize;
                indices[padded_flat] = Some(rem as u16);
            }
        }
        self.push_one_hot(name, indices, ROUND_LUT_LEN);
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

        let padded_dims = power_of_two_dims(logical_shape);
        let padded_len = padded_dims.iter().product::<usize>();
        let mut indices = vec![Some(padding_index as u16); padded_len];

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
            indices[padded_flat] = Some(entry as u16);
        }

        self.push_one_hot(name, indices, padded_entries);
    }

    fn push_one_hot(
        &mut self,
        name: impl Into<String>,
        nonzero_indices: Vec<Option<u16>>,
        entries: usize,
    ) {
        debug_assert!(entries.is_power_of_two());
        debug_assert!(nonzero_indices.len().is_power_of_two());
        self.entries.push(LayerPolynomialEntry {
            name: name.into(),
            poly: MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                nonzero_indices,
                entries,
            )),
        });
    }
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

fn power_of_two_dims(dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|dim| dim.next_power_of_two()).collect()
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
    hidden_state_commitments: HiddenStateCommitments<PCS::Commitment>,
    setup: &PCS::ProverSetup,
) -> LayerCommitments<PCS::Commitment>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut commitments = Vec::with_capacity(polynomials.entries.len());
    for entry in &polynomials.entries {
        let commitment = match entry.name.as_str() {
            "hidden_in" => hidden_state_commitments.hidden_in.clone(),
            "hidden_out" => hidden_state_commitments.hidden_out.clone(),
            _ => {
                let (commitment, _) = PCS::commit(&entry.poly, setup);
                commitment
            }
        };
        commitments.push(LayerCommitmentEntry {
            name: entry.name.clone(),
            commitment,
        });
    }
    LayerCommitments {
        entries: commitments,
    }
}

pub(crate) fn commit_layer_tensors<F, PCS>(
    hidden_out: &[i32],
    witness: &LayerWitness,
    hidden_state_commitments: HiddenStateCommitments<PCS::Commitment>,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    setup: &PCS::ProverSetup,
) -> LayerCommitmentsAndPolynomials<F, PCS::Commitment>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    // Build the committed polynomial set in one place so `prover.rs` does not
    // need to know which witness tensors are opened later by individual ops.
    let polynomials = LayerPolynomialMap::from_layer(hidden_out, witness, weights, shape, tensors);
    let commitments =
        commit_layer_polynomials::<F, PCS>(&polynomials, hidden_state_commitments, setup);
    LayerCommitmentsAndPolynomials {
        polynomials,
        commitments,
    }
}

pub fn absorb_layer_commitments<T, C>(
    transcript: &mut T,
    layer: usize,
    shape: &LayerShape,
    commitments: &LayerCommitments<C>,
) where
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
    transcript.append_u64(commitments.entries.len() as u64);
    for entry in &commitments.entries {
        transcript.append_bytes(entry.name.as_bytes());
        entry.commitment.append_to_transcript(transcript);
    }
}
