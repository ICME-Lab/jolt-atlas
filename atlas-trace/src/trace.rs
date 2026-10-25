//! The execution trace: the prover-facing output of running a model.

use std::{collections::BTreeMap, ops::Index};

use serde::{Deserialize, Serialize};

use crate::{
    id::{NodeId, Scale, SemVersion},
    tensor::Tensor,
};

/// The execution trace: one output tensor per node (inputs and constants
/// included), plus sparse per-node witnesses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    /// Output tensor per node, indexed by [`NodeId`].
    pub node_outputs: Vec<Tensor<i32>>,
    /// Sparse auxiliary witnesses (fused intermediates, softmax).
    pub aux: BTreeMap<NodeId, NodeAux>,
    /// The scale the trace was produced at.
    pub scale: Scale,
    /// The semantics version the trace was produced under.
    pub version: SemVersion,
}

/// Operands and output of a single node, as traced.
#[derive(Debug, Clone)]
pub struct LayerData<'a> {
    /// The node's output tensor.
    pub output: &'a Tensor<i32>,
    /// The node's operand tensors, in operand order.
    pub operands: Vec<&'a Tensor<i32>>,
}

/// Inputs and outputs of a model execution (the public IO the verifier
/// binds). Tensors are as stored in the trace: padded, uncropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionIO {
    /// Input tensors, in graph-input order.
    pub inputs: Vec<Tensor<i32>>,
    /// Output tensors, in graph-output order.
    pub outputs: Vec<Tensor<i32>>,
    /// Node indices of the inputs.
    pub input_indices: Vec<usize>,
    /// Node indices of the outputs.
    pub output_indices: Vec<usize>,
}

impl Trace {
    /// The output tensor of a node.
    pub fn output(&self, id: NodeId) -> &Tensor<i32> {
        &self.node_outputs[id.index()]
    }

    /// Operands and output of the node `id` (operands resolved through the
    /// node's input ids).
    pub fn layer_data<'a>(&'a self, id: NodeId, node: &crate::graph::Node) -> LayerData<'a> {
        LayerData {
            output: self.output(id),
            operands: self.operand_tensors(node),
        }
    }

    /// The operand tensors feeding `node`, in operand order.
    pub fn operand_tensors(&self, node: &crate::graph::Node) -> Vec<&Tensor<i32>> {
        node.inputs
            .iter()
            .map(|&id| &self.node_outputs[id.index()])
            .collect()
    }

    /// Operand tensors with per-dimension power-of-two padding applied.
    pub fn padded_operand_tensors(&self, node: &crate::graph::Node) -> Vec<Tensor<i32>> {
        self.operand_tensors(node)
            .into_iter()
            .map(Tensor::padded_pow2)
            .collect()
    }

    /// The public IO of this execution.
    pub fn io(&self, model: &crate::model::Model) -> ModelExecutionIO {
        let graph = model.graph();
        ModelExecutionIO {
            inputs: graph
                .inputs
                .iter()
                .map(|&id| self.node_outputs[id.index()].clone())
                .collect(),
            outputs: graph
                .outputs
                .iter()
                .map(|&id| self.node_outputs[id.index()].clone())
                .collect(),
            input_indices: graph.inputs.iter().map(|id| id.index()).collect(),
            output_indices: graph.outputs.iter().map(|id| id.index()).collect(),
        }
    }

    /// The fused-rescale intermediates of a node, if it is a fused op.
    pub fn fused(&self, id: NodeId) -> Option<&FusedIntermediates> {
        match self.aux.get(&id)? {
            NodeAux::Fused(f) => Some(f),
            NodeAux::Softmax(_) => None,
        }
    }

    /// The softmax witness of a node, if it is a softmax op.
    pub fn softmax(&self, id: NodeId) -> Option<&SoftmaxWitness> {
        match self.aux.get(&id)? {
            NodeAux::Softmax(w) => Some(w),
            NodeAux::Fused(_) => None,
        }
    }
}

/// Intermediates of a fused-rescale op (einsum, mul, square, cube,
/// mean-of-squares), captured so the proof system never re-runs the
/// expensive accumulation: `acc = quotient·D + remainder` with
/// `remainder ∈ [0, D)` and `output = sat_clamp_i32(quotient)`.
///
/// Same contract as the old crate's `FusedIntermediates`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusedIntermediates {
    /// Pre-clamp rescaled accumulation `acc div D` (the saturating-clamp
    /// lookup index).
    pub quotient: Tensor<i64>,
    /// Rescaling remainder `acc mod D ∈ [0, D)`.
    pub remainder: Tensor<i32>,
}

/// Full witness of an integer softmax node, captured in the trace so the
/// prover does not recompute the decomposed pipeline (an improvement over
/// the old crate, which recomputed it prover-side).
///
/// Layout: `rows = len / last_dim` row-major rows of width `last_dim`
/// (the padded width); lanes at `active_len..last_dim` are masked — every
/// per-lane witness value there is 0 and contributes nothing to row sums.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftmaxWitness {
    /// The multiplier `S = 2^scale`.
    pub scale: i64,
    /// Row width (padded last-axis size).
    pub last_dim: usize,
    /// Live lanes per row; lanes beyond this are masked padding.
    pub active_len: usize,
    /// Input logits, flat `[rows·last_dim]`.
    pub x: Vec<i32>,
    /// Per-row max over the live lanes, `[rows]`.
    pub max_k: Vec<i32>,
    /// Position of the max within each row's live lanes, `[rows]`.
    pub argmax_k: Vec<usize>,
    /// `exp_q[k,j] = ⌊LUT_hi[z_hi]·LUT_lo[z_lo]/S⌋` (0 on masked lanes).
    pub exp_q: Vec<i32>,
    /// `Σ_j exp_q[k,j]` over live lanes, `[rows]`.
    pub exp_sum_q: Vec<i32>,
    /// `⌊S² / exp_sum_q[k]⌋`, `[rows]`.
    pub inv_sum: Vec<i32>,
    /// `R[k,j] = exp_q·inv_sum − softmax_q·S ∈ [0, S)` (0 on masked lanes).
    pub r: Vec<i32>,
    /// High digit of the saturated centered logit (0 on masked lanes).
    pub z_hi: Vec<i32>,
    /// Low digit of the saturated centered logit (0 on masked lanes).
    pub z_lo: Vec<i32>,
    /// `LUT_hi[z_hi]` (0 on masked lanes).
    pub exp_hi: Vec<i32>,
    /// `LUT_lo[z_lo]` (0 on masked lanes).
    pub exp_lo: Vec<i32>,
    /// `exp_hi·exp_lo − exp_q·S ∈ [0, S)` (0 on masked lanes).
    pub r_exp: Vec<i32>,
}

/// Per-node auxiliary witness data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeAux {
    /// Fused-rescale quotient and remainder.
    Fused(FusedIntermediates),
    /// Softmax witness.
    Softmax(Box<SoftmaxWitness>),
}

impl Index<NodeId> for Trace {
    type Output = Tensor<i32>;

    fn index(&self, id: NodeId) -> &Self::Output {
        &self.node_outputs[id.index()]
    }
}
