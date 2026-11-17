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
            _ => None,
        }
    }

    /// The floor-division remainder of a node, if it is a `ScalarConstDiv`.
    pub fn remainder(&self, id: NodeId) -> Option<&Tensor<i32>> {
        match self.aux.get(&id)? {
            NodeAux::Remainder(r) => Some(r),
            _ => None,
        }
    }

    /// The division and square-root advice of a node, if it is an `Rsqrt`.
    pub fn rsqrt(&self, id: NodeId) -> Option<&RsqrtAdvice> {
        match self.aux.get(&id)? {
            NodeAux::Rsqrt(a) => Some(a),
            _ => None,
        }
    }

    /// The period reduction of a node's input, if it is a `Sin` or a `Cos`.
    pub fn trig_period(&self, id: NodeId) -> Option<&TrigPeriodAdvice> {
        match self.aux.get(&id)? {
            NodeAux::TrigPeriod(a) => Some(a),
            _ => None,
        }
    }

    /// The softmax witness of a node, if it is a softmax op.
    pub fn softmax(&self, id: NodeId) -> Option<&SoftmaxWitness> {
        match self.aux.get(&id)? {
            NodeAux::Softmax(w) => Some(w),
            _ => None,
        }
    }
}

/// Intermediates of a fused-rescale op (einsum, mul, square, cube,
/// mean-of-squares), captured so the proof system never re-runs the
/// expensive accumulation: `acc = output·D + remainder` with `remainder ∈ [0, D)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusedIntermediates {
    /// Rescaling remainder `acc mod D ∈ [0, D)`.
    pub remainder: Tensor<i32>,
}

/// The log-partition softmax's witness ([`crate::semantics::softmax`]).
///
/// Layout: `rows = len / last_dim` row-major rows of width `last_dim` (the padded width); lanes
/// at `active_len..last_dim` are padding, excluded from the row sums and zero in the output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftmaxWitness {
    /// Row width (padded last-axis size).
    pub last_dim: usize,
    /// Live lanes per row; lanes beyond this are padding.
    pub active_len: usize,
    /// The row's shift `c`, `[rows]`.
    pub shift: Vec<i32>,
    /// `u[k,j] = c[k] − x[k,j]`, on every lane of the padded row, `[rows·last_dim]`.
    pub address: Vec<i32>,
    /// `S' − Σ_j T_sum[ũ[k,j]]` over the live lanes, `[rows]`: non-negative.
    pub sum_gap: Vec<i64>,
    /// `Σ_j T_sum⁻[ũ[k,j]] − S' − 1` over the live lanes, `[rows]`: non-negative.
    pub prev_sum_gap: Vec<i64>,
}

/// Per-node auxiliary witness data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeAux {
    /// Fused-rescale remainder (the quotient is the node's output).
    Fused(FusedIntermediates),
    /// Softmax witness.
    Softmax(Box<SoftmaxWitness>),
    /// The remainder of a floor division (`ScalarConstDiv`): `a = out·b + remainder`,
    /// with the sign of the divisor.
    Remainder(Tensor<i32>),
    /// The division and square-root advice of an `Rsqrt`.
    Rsqrt(RsqrtAdvice),
    /// The period reduction of a `Sin` or `Cos` input.
    TrigPeriod(TrigPeriodAdvice),
}

/// The intermediates of `rsqrt(x) = ⌊√⌊S³ / x⌋⌋` for `x > 0`:
/// `S³ = quotient·x + div_remainder` with `div_remainder ∈ [0, x)`, and
/// `quotient = out² + sqrt_remainder` with `sqrt_remainder ∈ [0, 2·out]`. All three are zero
/// where `x ≤ 0` (the output is 0 there).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsqrtAdvice {
    /// `⌊S³ / x⌋`.
    pub quotient: Tensor<i64>,
    /// `S³ mod x`.
    pub div_remainder: Tensor<i32>,
    /// `quotient − out²`.
    pub sqrt_remainder: Tensor<i32>,
}

/// The Euclidean reduction of a trig input by the period modulus `P`:
/// `x = P·quotient + remainder` with `remainder ∈ [0, P)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrigPeriodAdvice {
    /// `x div P`.
    pub quotient: Tensor<i32>,
    /// `x mod P`, the value the table index is taken from.
    pub remainder: Tensor<i32>,
}

impl Index<NodeId> for Trace {
    type Output = Tensor<i32>;

    fn index(&self, id: NodeId) -> &Self::Output {
        &self.node_outputs[id.index()]
    }
}
