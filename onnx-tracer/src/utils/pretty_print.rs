use crate::{
    model::{ComputationGraph, Model},
    node::ComputationNode,
    ops::Operator,
};
use tabled::{
    Table, Tabled,
    settings::{Alignment, Modify, Style, object::Rows},
};

#[derive(Tabled)]
struct NodeRow {
    #[tabled(rename = "Node ID")]
    idx: usize,
    #[tabled(rename = "Operator")]
    operator: String,
    #[tabled(rename = "Details")]
    details: String,
    #[tabled(rename = "Inputs")]
    inputs: String,
    #[tabled(rename = "Output Dims")]
    output_dims: String,
}

impl From<&ComputationNode> for NodeRow {
    fn from(node: &ComputationNode) -> Self {
        let operator = format!("{:?}", node.operator);
        let operator = operator.split('(').next().unwrap_or(&operator).to_string();

        // Extract operator-specific details
        let details = match &node.operator {
            Operator::Clamp(op) => format!("axes: {}, spread: {}", op.axes, op.max_spread),
            Operator::Einsum(op) => format!("eq: {}", op.equation),
            Operator::Gather(op) => format!("axis: {}", op.axis),
            Operator::SoftmaxAxes(op) => format!("axes: {}", op.axes),
            Operator::Sum(op) => format!("axes: {:?}", op.axes),
            Operator::MoveAxis(op) => format!("src: {} → dst: {}", op.source, op.destination),
            Operator::Reshape(op) => format!("shape: {:?}", op.shape),
            Operator::Broadcast(op) => format!("shape: {:?}", op.shape),
            Operator::Rsqrt(op) => format!("scale: {}", op.scale),
            Operator::Tanh(op) => format!("scale: {}", op.scale),
            _ => "-".to_string(),
        };

        let inputs = if node.inputs.is_empty() {
            "-".to_string()
        } else {
            node.inputs
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        let output_dims = node
            .output_dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" × ");

        NodeRow {
            idx: node.idx,
            operator,
            details,
            inputs,
            output_dims,
        }
    }
}

impl Model {
    /// Returns a pretty-printed table representation of the model's computation graph
    pub fn pretty_print(&self) -> String {
        self.graph.pretty_print()
    }

    /// Returns an estimate of the total number of parameters in the model.
    ///
    /// This counts the elements in all `Constant` nodes, which represent
    /// weights, biases, embeddings, and other learned parameters.
    pub fn estimate_num_params(&self) -> usize {
        self.graph
            .nodes
            .values()
            .filter_map(|node| match &node.operator {
                Operator::Constant(tensor) => Some(tensor.0.len()),
                _ => None,
            })
            .sum()
    }

    /// Returns a human-readable string of the estimated parameter count (e.g., "1.2M", "350K").
    pub fn pretty_num_params(&self) -> String {
        let n = self.estimate_num_params();
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            format!("{n}")
        }
    }
}

impl ComputationGraph {
    /// Returns a pretty-printed table representation of the computation graph
    pub fn pretty_print(&self) -> String {
        let mut output = String::new();

        // Count parameters from Constant nodes
        let num_params: usize = self
            .nodes
            .values()
            .filter_map(|node| match &node.operator {
                Operator::Constant(tensor) => Some(tensor.0.len()),
                _ => None,
            })
            .sum();

        let params_str = if num_params >= 1_000_000_000 {
            format!("{:.2}B", num_params as f64 / 1_000_000_000.0)
        } else if num_params >= 1_000_000 {
            format!("{:.2}M", num_params as f64 / 1_000_000.0)
        } else if num_params >= 1_000 {
            format!("{:.1}K", num_params as f64 / 1_000.0)
        } else {
            format!("{num_params}")
        };

        // Add graph summary
        output.push_str("Computation Graph Summary\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output.push_str(&format!("Total Nodes: {}\n", self.nodes.len()));
        output.push_str(&format!(
            "Estimated Parameters: {num_params} ({params_str})\n"
        ));
        output.push_str(&format!("Input Nodes: {:?}\n", self.inputs));
        output.push_str(&format!("Output Nodes: {:?}\n\n", self.outputs));

        // Convert nodes to rows
        let rows: Vec<NodeRow> = self.nodes.values().map(NodeRow::from).collect();

        if rows.is_empty() {
            output.push_str("No nodes in graph.\n");
            return output;
        }

        // Create and style the table
        let table = Table::new(rows)
            .with(Style::modern())
            .with(Modify::new(Rows::first()).with(Alignment::center()))
            .to_string();

        output.push_str(&table);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Operator;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn test_pretty_print_empty_graph() {
        let graph = ComputationGraph {
            nodes: BTreeMap::new(),
            inputs: vec![],
            outputs: vec![],
            original_input_dims: HashMap::new(),
            original_output_dims: HashMap::new(),
        };
        let output = graph.pretty_print();
        assert!(output.contains("No nodes in graph"));
    }

    #[test]
    fn test_pretty_print_simple_graph() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            0,
            ComputationNode {
                idx: 0,
                operator: Operator::Input(Default::default()),
                inputs: vec![],
                output_dims: vec![1, 2],
            },
        );
        nodes.insert(
            1,
            ComputationNode {
                idx: 1,
                operator: Operator::Add(Default::default()),
                inputs: vec![0],
                output_dims: vec![1, 2],
            },
        );

        let graph = ComputationGraph {
            nodes,
            inputs: vec![0],
            outputs: vec![1],
            original_input_dims: HashMap::new(),
            original_output_dims: HashMap::new(),
        };

        let output = graph.pretty_print();
        assert!(output.contains("Node ID"));
        assert!(output.contains("Operator"));
        assert!(output.contains("Input"));
        assert!(output.contains("Add"));
    }
}
