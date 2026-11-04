//! The executor: run a compiled [`Model`] over quantized inputs and produce
//! a [`Trace`].

use std::collections::BTreeMap;

use crate::{
    error::ExecError,
    graph::Node,
    id::NodeId,
    model::Model,
    op::{Eval, EvalCtx, Op},
    tensor::Tensor,
    trace::Trace,
};

impl Model {
    /// Execute the model, producing the full per-node trace.
    ///
    /// `inputs` are quantized tensors in the graph's input order, with the
    /// model's **original** (unpadded) dims; the executor pads them.
    pub fn trace(&self, inputs: &[Tensor<i32>]) -> Result<Trace, ExecError> {
        let graph = self.graph();
        let mut padded_inputs = self.pad_inputs(inputs)?;
        let ctx = self.eval_ctx();

        let mut outputs: Vec<Tensor<i32>> = Vec::with_capacity(graph.nodes.len());
        let mut aux = BTreeMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            let id = NodeId(i as u32);
            let output = if let Op::Input(_) = node.op {
                padded_inputs
                    .remove(&id)
                    .expect("validated against graph.inputs")
            } else {
                let eval = eval_node(id, node, &outputs, &ctx)?;
                if let Some(a) = eval.aux {
                    aux.insert(id, a);
                }
                eval.output
            };
            check_output_dims(id, node, &self.physical_dims(id), &output);
            outputs.push(output);
        }

        Ok(Trace {
            node_outputs: outputs,
            aux,
            scale: self.scale(),
            version: self.version(),
        })
    }

    /// Validate `inputs` against the model's raw input dims and pad each to
    /// its Input node's physical dims, keyed by that node's id.
    fn pad_inputs(
        &self,
        inputs: &[Tensor<i32>],
    ) -> Result<BTreeMap<NodeId, Tensor<i32>>, ExecError> {
        let graph = self.graph();
        if inputs.len() != graph.inputs.len() {
            return Err(ExecError::WrongInputCount {
                expected: graph.inputs.len(),
                got: inputs.len(),
            });
        }
        let mut padded_inputs = BTreeMap::new();
        let raw_input_dims = self.raw_input_dims();
        for (i, (tensor, &node_id)) in inputs.iter().zip(&graph.inputs).enumerate() {
            let expected = raw_input_dims[i];
            if tensor.dims() != expected {
                return Err(ExecError::InputShape {
                    index: i,
                    expected: expected.to_vec(),
                    got: tensor.dims().to_vec(),
                });
            }
            let target = self.physical_dims(node_id);
            let padded = tensor.pad_to(&target).expect("padded dims grow every dim");
            padded_inputs.insert(node_id, padded);
        }
        Ok(padded_inputs)
    }

    fn eval_ctx(&self) -> EvalCtx<'_> {
        EvalCtx {
            scale: self.scale(),
            version: self.version(),
            tables: &self.tables,
            layout: self.layout(),
        }
    }

    /// Execute and return just the graph outputs, cropped back to their raw
    /// dims.
    pub fn forward(&self, inputs: &[Tensor<i32>]) -> Result<Vec<Tensor<i32>>, ExecError> {
        let trace = self.trace(inputs)?;
        Ok(self
            .graph()
            .outputs
            .iter()
            .zip(self.raw_output_dims())
            .map(|(&id, raw)| {
                trace
                    .output(id)
                    .crop_to(raw)
                    .expect("raw dims never exceed physical dims")
            })
            .collect())
    }
}

/// Evaluate one non-input node against the outputs of the nodes before it.
fn eval_node(
    id: NodeId,
    node: &Node,
    outputs: &[Tensor<i32>],
    ctx: &EvalCtx<'_>,
) -> Result<Eval, ExecError> {
    let operand_refs: Vec<&Tensor<i32>> = node
        .inputs
        .iter()
        .map(|&input| &outputs[input.index()])
        .collect();
    node.op
        .spec()
        .eval(&operand_refs, ctx)
        .map_err(|source| ExecError::Kernel {
            node: id,
            op: node.op.to_string(),
            source,
        })
}

/// The kernel's output must land on the node's physical dims; a mismatch is
/// an internal invariant violation, not a user error.
fn check_output_dims(id: NodeId, node: &Node, physical: &[usize], output: &Tensor<i32>) {
    assert_eq!(
        output.dims(),
        physical,
        "node {id} ({}) produced {:?}, declared {:?}",
        node.op,
        output.dims(),
        physical
    );
}
