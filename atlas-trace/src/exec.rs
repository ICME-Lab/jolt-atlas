//! The executor: run a compiled [`Model`] over quantized inputs and produce
//! a [`Trace`].

use std::collections::BTreeMap;

use crate::{
    error::ExecError,
    id::NodeId,
    model::Model,
    op::{EvalCtx, Op},
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
        if inputs.len() != graph.inputs.len() {
            return Err(ExecError::WrongInputCount {
                expected: graph.inputs.len(),
                got: inputs.len(),
            });
        }
        // Validate + pad the supplied inputs, keyed by Input node id.
        let mut padded_inputs: BTreeMap<NodeId, Tensor<i32>> = BTreeMap::new();
        for (i, (tensor, &node_id)) in inputs.iter().zip(&graph.inputs).enumerate() {
            let expected = &self.io_dims().inputs[i];
            if tensor.dims() != expected.as_slice() {
                return Err(ExecError::InputShape {
                    index: i,
                    expected: expected.clone(),
                    got: tensor.dims().to_vec(),
                });
            }
            let target = &graph.node(node_id).output_dims;
            let padded = tensor.pad_to(target).expect("padded dims grow every dim");
            padded_inputs.insert(node_id, padded);
        }

        let ctx = EvalCtx {
            scale: self.scale(),
            version: self.version(),
            tables: &self.tables,
        };

        let mut outputs: Vec<Tensor<i32>> = Vec::with_capacity(graph.nodes.len());
        let mut aux = BTreeMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            let id = NodeId(i as u32);
            let output = if let Op::Input(_) = node.op {
                padded_inputs
                    .remove(&id)
                    .expect("validated against graph.inputs")
            } else {
                let operand_refs: Vec<&Tensor<i32>> = node
                    .inputs
                    .iter()
                    .map(|&input| &outputs[input.index()])
                    .collect();
                let eval = node
                    .op
                    .spec()
                    .eval(&operand_refs, &ctx)
                    .map_err(|source| ExecError::Kernel {
                        node: id,
                        op: node.op.to_string(),
                        source,
                    })?;
                if let Some(a) = eval.aux {
                    aux.insert(id, a);
                }
                eval.output
            };
            // The kernel's output must land on the declared dims; a mismatch
            // is an internal invariant violation, not a user error.
            assert_eq!(
                output.dims(),
                node.output_dims.as_slice(),
                "node {id} ({}) produced {:?}, declared {:?}",
                node.op,
                output.dims(),
                node.output_dims
            );
            outputs.push(output);
        }

        Ok(Trace {
            node_outputs: outputs,
            aux,
            scale: self.scale(),
            version: self.version(),
        })
    }

    /// Execute and return just the graph outputs, cropped back to their
    /// original (unpadded) dims.
    pub fn forward(&self, inputs: &[Tensor<i32>]) -> Result<Vec<Tensor<i32>>, ExecError> {
        let trace = self.trace(inputs)?;
        Ok(self
            .graph()
            .outputs
            .iter()
            .zip(&self.io_dims().outputs)
            .map(|(&id, original)| {
                trace
                    .output(id)
                    .crop_to(original)
                    .expect("original dims never exceed padded dims")
            })
            .collect())
    }
}
