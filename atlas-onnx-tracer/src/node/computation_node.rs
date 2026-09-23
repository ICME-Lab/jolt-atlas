//! The `ComputationNode` type.
//!
//! Kept in its own module so that `node::handlers` is a sibling rather than a
//! descendant, and therefore cannot reach [`ComputationNode::output_dims`]
//! directly.
use crate::ops::Operator;
use crate::utils::dims::UsizeDimsExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Represents a single computation node in the computation graph.
///
/// Nodes carry their operator, input dependencies (by index), and the
/// output tensor dimensions produced by the operator.
pub struct ComputationNode {
    /// Stable node index within the graph (0-based).
    pub idx: usize,
    /// The operation executed by this node.
    pub operator: Operator,
    /// Indices of upstream nodes whose outputs feed this node.
    pub inputs: Vec<usize>,
    /// Dimensions (shape) of the tensor produced by this node.
    ///
    /// Read through [`ComputationNode::raw_or_padded_output_dims`].
    output_dims: Vec<usize>,
    /// Address width (bits) of this node's saturating-clamp lookup, if it has
    /// one: the two's-complement width that provably holds the pre-clamp value.
    /// `64` unless narrowed by [`Model::annotate_clamp_widths`]
    /// (see `model::clamp_width`).
    ///
    /// [`Model::annotate_clamp_widths`]: crate::model::Model::annotate_clamp_widths
    #[serde(default = "default_sat_clamp_bits")]
    pub sat_clamp_bits: usize,
}

fn default_sat_clamp_bits() -> usize {
    crate::model::clamp_width::CLAMP_WIDTH_MAX
}

impl ComputationNode {
    /// Construct a new computation node.
    ///
    /// - `idx`: Stable index of the node within the graph.
    /// - `operator`: The operator this node performs.
    /// - `inputs`: Indices of nodes providing inputs to this node.
    /// - `output_dims`: Shape of the output tensor produced.
    pub fn new(
        idx: usize,
        operator: Operator,
        inputs: Vec<usize>,
        output_dims: Vec<usize>,
    ) -> Self {
        Self {
            idx,
            operator,
            inputs,
            output_dims,
            sat_clamp_bits: default_sat_clamp_bits(),
        }
    }

    /// Computes the total number of output elements produced by this node
    /// after mapping each output dimension to its next power of two.
    ///
    /// For example, if `output_dims` is `[2, 3]`, this returns `8`
    /// because dimensions are normalized to `[2, 4]` before taking the product.
    pub fn pow2_padded_num_output_elements(&self) -> usize {
        self.output_dims
            .map_next_power_of_two()
            .into_iter()
            .product()
    }

    /// Computes the total number of output elements produced by this node
    /// without applying power-of-two padding.
    pub fn num_output_elements(&self) -> usize {
        self.output_dims.iter().product()
    }

    /// This node's output dimensions exactly as stored, raw or padded.
    ///
    /// The name is the honest one, because which of the two you get depends on
    /// where the caller sits in the loading pipeline. Nodes are built with the
    /// shape the ONNX graph declares, and [`ModelLoader::pad`] later rounds
    /// every stored dimension up to a power of two in place — and only when
    /// [`RunArgs::pad_to_power_of_2`] is set. Callers that run before that
    /// pass, such as the operator handlers, the original-dimension capture and
    /// the reshape planner, observe raw dimensions; callers after it observe
    /// padded ones, unless padding was never requested.
    ///
    /// No caller should have to care. Each call site is to be resolved to a
    /// definite raw or padded reading, at which point this accessor goes away.
    ///
    /// [`ModelLoader::pad`]: crate::model::load::ModelLoader::pad
    /// [`RunArgs::pad_to_power_of_2`]: crate::model::RunArgs::pad_to_power_of_2
    pub fn raw_or_padded_output_dims(&self) -> Vec<usize> {
        self.output_dims.clone()
    }

    /// Returns true if the output of this node is a scalar (i.e., has exactly one element).
    pub fn is_scalar(&self) -> bool {
        self.num_output_elements() == 1
    }

    /// Rounds every stored output dimension up to the next power of two.
    pub(crate) fn pad_output_dims_to_power_of_2(&mut self) {
        self.output_dims = self.output_dims.map_next_power_of_two();
    }
}
