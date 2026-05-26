use ark_bn254::{Bn254, Fr};
use common::CommittedPoly;
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProverKey},
        },
        multilinear_polynomial::MultilinearPolynomial,
        one_hot_polynomial::OneHotPolynomial,
    },
    subprotocols::shout::compute_instruction_h_indices,
    transcripts::{AppendToTranscript, Transcript},
};

#[cfg(test)]
use crate::claim::Claim;
use crate::{
    claim::Poly,
    ops::round::ROUND_LUT_LEN,
    streaming_srs::{FlatG1SrsReader, StreamingOneHotCommitter},
};

use super::{
    polys::LayerPolys,
    tensors::{LayerTensorIds, round_site},
    types::{LayerShape, LayerWeights},
    witness::LayerWitness,
};

// Commitments for one layer. This file answers what is committed and how those
// commitments are bound into the transcript. The PCS opening proof is
// in `openings.rs`.

#[derive(Debug, Clone)]
pub struct LayerPolySetEntry<F: JoltField> {
    pub name: String,
    pub committed_poly: CommittedPoly,
    pub poly: MultilinearPolynomial<F>,
}

#[derive(Debug, Clone)]
pub struct CommittedLayerPolyEntry<F: JoltField, C> {
    pub name: String,
    // Adapter-only handle for the current core PCS/opening-reduction API.
    // The claim flow should use `poly` below; this handle disappears once the
    // core accumulator accepts owned polys directly.
    pub committed_poly: CommittedPoly,
    pub poly: Poly<F, C>,
}

#[derive(Debug, Clone, Default)]
pub struct CommittedLayerPolys<F: JoltField, C> {
    pub entries: Vec<CommittedLayerPolyEntry<F, C>>,
}

#[derive(Debug, Clone, Default)]
pub struct LayerPolySet<F: JoltField> {
    pub entries: Vec<LayerPolySetEntry<F>>,
}

#[derive(Debug, Clone)]
pub struct LayerCommitmentEntry<C> {
    pub name: String,
    pub committed_poly: CommittedPoly,
    pub commitment: C,
}

#[derive(Debug, Clone, Default)]
pub struct LayerCommitments<C> {
    pub entries: Vec<LayerCommitmentEntry<C>>,
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
                    committed_poly: entry.committed_poly,
                    commitment,
                }
            })
            .collect();
        Self { entries }
    }
}

impl<F: JoltField> LayerPolySet<F> {
    pub fn from_layer(
        witness: &LayerWitness,
        _weights: &LayerWeights,
        shape: &LayerShape,
        tensors: &LayerTensorIds,
    ) -> Self {
        let mut out = Self::default();

        // Commit only RA/lookup polynomials created inside this layer.
        // Hidden state commitments are supplied by the caller, and ordinary
        // witness tensors/accumulators are consumed by op sumchecks.

        out.push_round_ra(
            format!("{}_round_ra", tensors.context_acc),
            &witness.context_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
            round_site::CONTEXT,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.o_proj_acc),
            &witness.o_proj_acc,
            &[shape.seq, shape.hidden],
            round_site::O_PROJ,
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
            "softmax_output_round_ra",
            &softmax_floor_as_i64,
            &[shape.q_heads, shape.seq, shape.seq],
            round_site::SOFTMAX_OUTPUT,
        );
        out.push_round_ra(
            "softmax_floor_round_ra",
            &softmax_acc_shifted,
            &[shape.q_heads, shape.seq, shape.seq],
            round_site::SOFTMAX_FLOOR,
        );
        out.push_round_ra(
            "softmax_exp_acc_round_ra",
            &witness.softmax_exp_acc,
            &[shape.q_heads, shape.seq, shape.seq],
            round_site::SOFTMAX_EXP,
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
            |d| CommittedPoly::QwenSoftmaxExpRaD(d),
        );
        out.push_softmax_input_frac_ra(
            "softmax_input_frac_ra",
            &witness.qk_score,
            &witness.softmax_max,
            shape.q_heads,
            shape.seq,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.qk_score_scale_acc),
            &witness.qk_score_scale_acc,
            &[shape.q_heads, shape.seq, shape.seq],
            round_site::QK_SCORE_SCALE,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.qk_score_acc),
            &witness.qk_score_acc,
            &[shape.q_heads, shape.seq, shape.seq],
            round_site::QK_SCORE_DOT,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.q_rope_acc),
            &witness.q_rope_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
            round_site::Q_ROPE,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_rope_acc),
            &witness.k_rope_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
            round_site::K_ROPE,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.q_proj_acc),
            &witness.q_proj_acc,
            &[shape.seq, shape.attention_width()],
            round_site::Q_PROJ,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_proj_acc),
            &witness.k_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
            round_site::K_PROJ,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.v_proj_acc),
            &witness.v_proj_acc,
            &[shape.seq, shape.kv_heads * shape.head_dim],
            round_site::V_PROJ,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.q_norm_acc),
            &witness.q_norm_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
            round_site::Q_NORM,
        );
        out.push_round_ra(
            "q_norm_norm_acc_round_ra",
            &witness.q_norm_norm_acc,
            &[shape.seq, shape.q_heads, shape.head_dim],
            round_site::Q_NORM_INTERNAL,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.k_norm_acc),
            &witness.k_norm_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
            round_site::K_NORM,
        );
        out.push_round_ra(
            "k_norm_norm_acc_round_ra",
            &witness.k_norm_norm_acc,
            &[shape.seq, shape.kv_heads, shape.head_dim],
            round_site::K_NORM_INTERNAL,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_atten_acc),
            &witness.rms_norm_atten_acc,
            &[witness.rms_norm_atten_a.len()],
            round_site::RMS_NORM_ATTEN,
        );
        out.push_round_ra(
            "rms_norm_atten_norm_acc_round_ra",
            &witness.rms_norm_atten_norm_acc,
            &[shape.seq, shape.hidden],
            round_site::RMS_NORM_ATTEN_INTERNAL,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.rms_norm_mlp_acc),
            &witness.rms_norm_mlp_acc,
            &[shape.seq, shape.hidden],
            round_site::RMS_NORM_MLP,
        );
        out.push_round_ra(
            "rms_norm_mlp_norm_acc_round_ra",
            &witness.rms_norm_mlp_norm_acc,
            &[shape.seq, shape.hidden],
            round_site::RMS_NORM_MLP_INTERNAL,
        );

        out.push_round_ra(
            format!("{}_round_ra", tensors.gate_proj_acc),
            &witness.gate_proj_acc,
            &[shape.seq, shape.intermediate],
            round_site::GATE_PROJ,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.up_proj_acc),
            &witness.up_proj_acc,
            &[shape.seq, shape.intermediate],
            round_site::UP_PROJ,
        );
        out.push_round_index_ra(
            "silu_round_ra",
            &witness
                .gate_proj
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>(),
            &[shape.seq, shape.intermediate],
            |d| CommittedPoly::QwenSiluRoundRaD(d),
        );
        out.push_round_index_ra(
            "silu_output_round_ra",
            &witness.silu_acc,
            &[shape.seq, shape.intermediate],
            |d| CommittedPoly::QwenSiluOutputRoundRaD(d),
        );
        out.push_lookup_ra(
            "silu_ra",
            &witness.silu_ra,
            &[shape.seq, shape.intermediate],
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            padded_silu_lut_len(silu_entries(witness.silu_min_n, witness.silu_max_n)),
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            |d| CommittedPoly::QwenSiluBaseRaD(d),
        );
        out.push_lookup_ra(
            "silu_slope_ra",
            &witness.silu_ra,
            &[shape.seq, shape.intermediate],
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            padded_silu_lut_len(silu_entries(witness.silu_min_n, witness.silu_max_n)),
            silu_entries(witness.silu_min_n, witness.silu_max_n),
            |d| CommittedPoly::QwenSiluSlopeRaD(d),
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.silu_up_acc),
            &witness.silu_up_acc,
            &[shape.seq, shape.intermediate],
            round_site::SILU_UP,
        );
        out.push_round_ra(
            format!("{}_round_ra", tensors.down_proj_acc),
            &witness.down_proj_acc,
            &[shape.seq, shape.hidden],
            round_site::DOWN_PROJ,
        );
        out
    }

    #[cfg(test)]
    pub fn entry_for_committed_poly(&self, poly: CommittedPoly) -> Option<&LayerPolySetEntry<F>> {
        self.entries
            .iter()
            .find(|entry| entry.committed_poly == poly)
    }

    #[cfg(test)]
    pub fn opening_value_mismatches<C>(&self, claims: &[Claim<F, C>]) -> Vec<String> {
        let mut mismatches = Vec::new();
        for (idx, claim) in claims.iter().enumerate() {
            let value = joltworks::poly::multilinear_polynomial::PolynomialEvaluation::evaluate(
                &claim.poly.data,
                &claim.point,
            );
            if value != claim.value {
                mismatches.push(format!("claim {idx}: value mismatch"));
            }
        }
        mismatches
    }

    fn push_round_ra(
        &mut self,
        name: impl Into<String>,
        input: &[i64],
        logical_shape: &[usize],
        lookup_site: usize,
    ) {
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
        self.push_one_hot_decomposition(name, indices, ROUND_LUT_LEN, |d| {
            CommittedPoly::QwenRoundRaD(lookup_site, d)
        });
    }

    fn push_round_index_ra(
        &mut self,
        name: impl Into<String>,
        input: &[i64],
        logical_shape: &[usize],
        committed_poly: impl Fn(usize) -> CommittedPoly,
    ) {
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
            indices[padded_flat] = Some(value.rem_euclid(ROUND_LUT_LEN as i64) as u16);
        }
        self.push_one_hot_decomposition(name, indices, ROUND_LUT_LEN, committed_poly);
    }

    fn push_softmax_input_frac_ra(
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
        self.push_one_hot_decomposition(name, indices, ROUND_LUT_LEN, |d| {
            CommittedPoly::QwenSoftmaxInputFracRaD(d)
        });
    }

    fn push_lookup_ra(
        &mut self,
        name: impl Into<String>,
        logical_onehot: &[u8],
        logical_shape: &[usize],
        logical_entries: usize,
        padded_entries: usize,
        padding_index: usize,
        committed_poly: impl Fn(usize) -> CommittedPoly,
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

        self.push_one_hot_decomposition(name, indices, padded_entries, committed_poly);
    }

    fn push_one_hot_decomposition(
        &mut self,
        name: impl Into<String>,
        nonzero_indices: Vec<Option<u16>>,
        entries: usize,
        committed_poly: impl Fn(usize) -> CommittedPoly,
    ) {
        debug_assert!(entries.is_power_of_two());
        debug_assert!(nonzero_indices.len().is_power_of_two());
        let one_hot_params = OneHotParams::from_config_and_log_K(
            &OneHotConfig::default(),
            entries.trailing_zeros() as usize,
        );
        let lookup_indices = nonzero_indices
            .into_iter()
            .map(|idx| idx.map(usize::from).unwrap_or(0))
            .collect::<Vec<_>>();
        let chunks = compute_instruction_h_indices(&lookup_indices, &one_hot_params);
        let name = name.into();
        for (d, chunk) in chunks.into_iter().enumerate() {
            self.entries.push(LayerPolySetEntry {
                name: format!("{name}.rad.{d}"),
                committed_poly: committed_poly(d),
                poly: MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    chunk.into_iter().map(|idx| idx.map(u16::from)).collect(),
                    one_hot_params.k_chunk,
                )),
            });
        }
    }
}

impl<F: JoltField, C: Clone> CommittedLayerPolys<F, C> {
    pub fn find(&self, name: &str) -> crate::Result<&CommittedLayerPolyEntry<F, C>> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| crate::ProverError::MissingCommittedPolynomials(vec![name.to_string()]))
    }
}

impl<F: JoltField> LayerPolySet<F> {
    pub fn attach_commitments<C: Clone>(
        self,
        commitments: &LayerCommitments<C>,
    ) -> crate::Result<CommittedLayerPolys<F, C>> {
        let mut entries = Vec::with_capacity(self.entries.len());
        for entry in self.entries {
            let Some(commitment) = commitments
                .entries
                .iter()
                .find(|candidate| candidate.committed_poly == entry.committed_poly)
                .map(|candidate| candidate.commitment.clone())
            else {
                return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                    format!("{:?}", entry.committed_poly),
                ]));
            };
            entries.push(CommittedLayerPolyEntry {
                name: entry.name,
                committed_poly: entry.committed_poly,
                poly: Poly::new(entry.poly, Some(commitment)),
            });
        }
        Ok(CommittedLayerPolys { entries })
    }
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
    poly_set: &LayerPolySet<F>,
    setup: &PCS::ProverSetup,
) -> LayerCommitments<PCS::Commitment>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut commitments = Vec::with_capacity(poly_set.entries.len());
    for entry in &poly_set.entries {
        let (commitment, _) = PCS::commit(&entry.poly, setup);
        commitments.push(LayerCommitmentEntry {
            name: entry.name.clone(),
            committed_poly: entry.committed_poly,
            commitment,
        });
    }
    LayerCommitments {
        entries: commitments,
    }
}

pub fn commit_layer_polynomials_streaming_onehot(
    poly_set: &LayerPolySet<Fr>,
    dense_setup: &HyperKZGProverKey<Bn254>,
    flat_srs: &FlatG1SrsReader,
    srs_chunk_len: usize,
    onehot_threads: Option<usize>,
    report_metrics: bool,
    sort_onehot_indices: bool,
) -> crate::Result<LayerCommitments<HyperKZGCommitment<Bn254>>> {
    let mut commitments = Vec::with_capacity(poly_set.entries.len());
    let mut onehot_positions = Vec::new();
    let mut onehot_committer =
        StreamingOneHotCommitter::with_threads(flat_srs.clone(), srs_chunk_len, onehot_threads)?
            .with_sorted_indices(sort_onehot_indices)
            .with_metrics(report_metrics);

    for entry in &poly_set.entries {
        match &entry.poly {
            MultilinearPolynomial::OneHot(one_hot) => {
                let poly = entry.committed_poly;
                onehot_committer.add_one_hot(poly, one_hot)?;
                onehot_positions.push((commitments.len(), poly));
                commitments.push(LayerCommitmentEntry {
                    name: entry.name.clone(),
                    committed_poly: entry.committed_poly,
                    commitment: HyperKZGCommitment::default(),
                });
            }
            _ => {
                let (commitment, _) = HyperKZG::<Bn254>::commit(&entry.poly, dense_setup);
                commitments.push(LayerCommitmentEntry {
                    name: entry.name.clone(),
                    committed_poly: entry.committed_poly,
                    commitment,
                });
            }
        }
    }

    let mut onehot_commitments = onehot_committer.commit_all()?;
    for (pos, poly) in onehot_positions {
        commitments[pos].commitment = onehot_commitments.remove(&poly).ok_or_else(|| {
            crate::ProverError::MissingCommittedPolynomials(vec![format!("{poly:?}")])
        })?;
    }

    Ok(LayerCommitments {
        entries: commitments,
    })
}

pub(crate) fn commit_layer_ra_polys<F, PCS>(
    polys: &mut LayerPolys<F, PCS::Commitment>,
    tensors: &LayerTensorIds,
    setup: &PCS::ProverSetup,
) -> LayerCommitments<PCS::Commitment>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut entries = Vec::new();

    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.context_acc),
        &mut polys.pv_matmul_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::CONTEXT, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.o_proj_acc),
        &mut polys.o_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::O_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "softmax_output_round_ra",
        &mut polys.softmax_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_OUTPUT, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "softmax_floor_round_ra",
        &mut polys.softmax_floor_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_FLOOR, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "softmax_exp_acc_round_ra",
        &mut polys.softmax_exp_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_EXP, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "softmax_ra",
        &mut polys.softmax_ra,
        CommittedPoly::QwenSoftmaxExpRaD,
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "softmax_input_frac_ra",
        &mut polys.softmax_input_frac_ra,
        CommittedPoly::QwenSoftmaxInputFracRaD,
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.qk_score_scale_acc),
        &mut polys.qk_score_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_SCALE, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.qk_score_acc),
        &mut polys.qk_score_dot_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_DOT, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.q_rope_acc),
        &mut polys.q_rope_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_ROPE, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.k_rope_acc),
        &mut polys.k_rope_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_ROPE, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.q_proj_acc),
        &mut polys.q_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.k_proj_acc),
        &mut polys.k_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.v_proj_acc),
        &mut polys.v_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::V_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.q_norm_acc),
        &mut polys.q_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_NORM, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "q_norm_norm_acc_round_ra",
        &mut polys.q_norm_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_NORM_INTERNAL, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.k_norm_acc),
        &mut polys.k_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_NORM, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "k_norm_norm_acc_round_ra",
        &mut polys.k_norm_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_NORM_INTERNAL, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.rms_norm_atten_acc),
        &mut polys.rms_norm_atten_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "rms_norm_atten_norm_acc_round_ra",
        &mut polys.rms_norm_atten_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN_INTERNAL, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.rms_norm_mlp_acc),
        &mut polys.rms_norm_mlp_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "rms_norm_mlp_norm_acc_round_ra",
        &mut polys.rms_norm_mlp_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP_INTERNAL, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.gate_proj_acc),
        &mut polys.gate_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::GATE_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.up_proj_acc),
        &mut polys.up_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::UP_PROJ, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "silu_round_ra",
        &mut polys.silu_gate_round_ra,
        CommittedPoly::QwenSiluRoundRaD,
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "silu_output_round_ra",
        &mut polys.silu_round_ra,
        CommittedPoly::QwenSiluOutputRoundRaD,
        setup,
    );
    let silu_split = polys.silu_ra.len() / 2;
    let (silu_base_ra, silu_slope_ra) = polys.silu_ra.split_at_mut(silu_split);
    commit_ra_group::<F, PCS>(
        &mut entries,
        "silu_ra",
        silu_base_ra,
        CommittedPoly::QwenSiluBaseRaD,
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        "silu_slope_ra",
        silu_slope_ra,
        CommittedPoly::QwenSiluSlopeRaD,
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.silu_up_acc),
        &mut polys.silu_up_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SILU_UP, d),
        setup,
    );
    commit_ra_group::<F, PCS>(
        &mut entries,
        format!("{}_round_ra", tensors.down_proj_acc),
        &mut polys.down_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::DOWN_PROJ, d),
        setup,
    );

    LayerCommitments { entries }
}

fn commit_ra_group<F, PCS>(
    entries: &mut Vec<LayerCommitmentEntry<PCS::Commitment>>,
    name: impl Into<String>,
    polys: &mut [Poly<F, PCS::Commitment>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
    setup: &PCS::ProverSetup,
) where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let name = name.into();
    for (d, poly) in polys.iter_mut().enumerate() {
        let (commitment, _) = PCS::commit(&poly.data, setup);
        poly.commitment = Some(commitment.clone());
        entries.push(LayerCommitmentEntry {
            name: format!("{name}.rad.{d}"),
            committed_poly: committed_poly(d),
            commitment,
        });
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
