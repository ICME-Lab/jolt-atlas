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
    /// Dimensions (shape) of the tensor produced by this node, as the ONNX
    /// graph declares them.
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

    /// Shape of this node's output tensor, as the ONNX graph declares it.
    pub fn raw_output_dims(&self) -> Vec<usize> {
        self.output_dims.clone()
    }

    /// Shape of this node's output tensor over the power-of-two domain the
    /// proof is defined on: every raw dimension rounded up to a power of two.
    pub fn padded_output_dims(&self) -> Vec<usize> {
        self.output_dims.map_next_power_of_two()
    }

    /// Returns true if the output of this node is a scalar (i.e., has exactly one element).
    pub fn is_scalar(&self) -> bool {
        self.num_output_elements() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::ComputationNode;
    use crate::ops::Operator;

    fn node(dims: &[usize]) -> ComputationNode {
        ComputationNode::new(
            0,
            Operator::Input(Default::default()),
            vec![],
            dims.to_vec(),
        )
    }

    #[test]
    fn padding_rounds_each_dimension_up_independently() {
        assert_eq!(node(&[3, 5]).padded_output_dims(), vec![4, 8]);
        assert_eq!(node(&[2, 3, 4]).padded_output_dims(), vec![2, 4, 4]);
        assert_eq!(node(&[7]).padded_output_dims(), vec![8]);
    }

    #[test]
    fn padding_leaves_powers_of_two_alone() {
        assert_eq!(node(&[4, 16]).padded_output_dims(), vec![4, 16]);
        assert_eq!(node(&[1, 1]).padded_output_dims(), vec![1, 1]);
        assert_eq!(node(&[]).padded_output_dims(), Vec::<usize>::new());
    }

    #[test]
    fn raw_dimensions_are_returned_unchanged() {
        assert_eq!(node(&[3, 5]).raw_output_dims(), vec![3, 5]);
        assert_eq!(node(&[4, 16]).raw_output_dims(), vec![4, 16]);
    }

    #[test]
    fn element_counts_follow_their_domain() {
        let n = node(&[3, 5]);
        assert_eq!(n.num_output_elements(), 15);
        assert_eq!(n.pow2_padded_num_output_elements(), 32);
    }
}
