//! Translate a supported Atlas integer model into registered native BlindFold relations.
//!
//! Model structure, constants, shapes and scales are public. Runtime inputs and
//! outputs remain hidden. This API proves the committed execution, not equality
//! to caller-supplied public IO. Use the native tensor boundary API to bind those
//! values to a public statement or another proof. The verifier must authenticate
//! the registration independently of the prover.
//!
//! This first port supports arithmetic, normalization and layouts on exact
//! power-of-two shapes. Implicit input padding and output cropping are rejected.
//! It does not replace the experimental HyperKZG dispatcher in [`super::zk`].

use atlas_onnx_tracer::{model::Model, node::ComputationNode, ops::Operator, tensor::Tensor};
use joltworks::{
    poly::commitment::dory::{
        native_graph::{NativeGraph, NativeGraphNode, NativeGraphStatement, NativeGraphWitness},
        native_registration::{NativeGraphPreprocessing, NativeRegisteredGraph},
        DoryProverSetup,
    },
    utils::errors::ProofVerifyError,
};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.into())
}

fn elements(shape: &[usize]) -> Result<usize, ProofVerifyError> {
    if shape.len() > 32 || shape.iter().any(|d| !d.is_power_of_two()) {
        return Err(invalid(
            "Native model dimensions must be positive powers of two",
        ));
    }
    let bits: u32 = shape.iter().map(|d| d.ilog2()).sum();
    if bits > 30 {
        return Err(invalid("Native model tensor exceeds the supported size"));
    }
    Ok(1usize << bits)
}

fn scale(value: i32) -> Result<u8, ProofVerifyError> {
    u8::try_from(value).map_err(|_| invalid("Invalid native model scale"))
}

fn append(graph: &mut NativeGraph, node: NativeGraphNode) -> usize {
    let id = graph.num_inputs() + graph.nodes.len();
    graph.nodes.push(node);
    id
}

// Ported from the application's graph assembler. Rebase every tensor operand,
// including auxiliary proof nodes after the designated numerical output.
fn inline(graph: &mut NativeGraph, small: NativeGraph, input: usize) -> usize {
    debug_assert_eq!(small.num_inputs(), 1);
    debug_assert_eq!(small.outputs.len(), 1);
    let first = graph.num_inputs() + graph.nodes.len();
    let map = |id: usize| if id == 0 { input } else { first + id - 1 };
    let output = map(small.outputs[0]);
    for mut node in small.nodes {
        node.input = map(node.input);
        if let Some(op) = &mut node.mul {
            op.right = map(op.right);
        }
        if let Some(op) = &mut node.add {
            op.right = map(op.right);
        }
        if let Some(op) = &mut node.einsum {
            op.right = map(op.right);
        }
        if let Some(op) = &mut node.concat {
            op.right = map(op.right);
        }
        if let Some(op) = &mut node.lookup {
            op.table_input = op.table_input.map(map);
        }
        graph.nodes.push(node);
    }
    output
}

/// An immutable translation of an expected public model, constructed without a
/// private witness. All runtime `Input` nodes are private; `Constant` nodes are
/// registered public parameters. The node map exposes original tensor identities
/// for subsequent boundary proofs.
pub struct NativeModel {
    graph: NativeGraph,
    tensors: BTreeMap<usize, usize>,
    inputs: Vec<usize>,
    constants: BTreeMap<usize, Vec<i32>>,
}

impl NativeModel {
    /// Translate an expected public model without executing it. The context
    /// identifies the caller's application or model registration. Unsupported
    /// operators, implicit padding and inconsistent graph metadata return errors.
    pub fn new(model: &Model, context: Vec<u8>) -> Result<Self, ProofVerifyError> {
        if context.is_empty() {
            return Err(invalid("A native model requires an authenticated context"));
        }
        let source = &model.graph;
        let declared_inputs: BTreeSet<_> = source.inputs.iter().copied().collect();
        let actual_inputs: BTreeSet<_> = source
            .nodes
            .iter()
            .filter_map(|(&id, node)| matches!(node.operator, Operator::Input(_)).then_some(id))
            .collect();
        if declared_inputs != actual_inputs || declared_inputs.len() != source.inputs.len() {
            return Err(invalid(
                "Native model input list must contain each Input node exactly once",
            ));
        }
        // Padding/cropping needs its own binding relation. Do not silently prove
        // a computation with different external IO semantics.
        for (metadata, external) in [
            (&source.original_input_dims, &source.inputs),
            (&source.original_output_dims, &source.outputs),
        ] {
            for (&id, dims) in metadata {
                if !external.contains(&id)
                    || source.nodes.get(&id).is_none_or(|n| n.output_dims != *dims)
                {
                    return Err(invalid(
                        "Implicit native model padding or cropping is unsupported",
                    ));
                }
            }
        }
        let mut graph = NativeGraph {
            context: b"atlas-native-model-v1".to_vec(),
            input_shapes: vec![],
            nodes: vec![],
            outputs: vec![],
        };
        graph
            .context
            .extend_from_slice(&(context.len() as u64).to_le_bytes());
        graph.context.extend_from_slice(&context);
        graph.context.extend_from_slice(&model.scale.to_le_bytes());
        let mut tensors = BTreeMap::new();
        let mut constants = BTreeMap::new();
        for (&id, node) in &source.nodes {
            if id != node.idx
                || node
                    .inputs
                    .iter()
                    .any(|i| *i >= id || !source.nodes.contains_key(i))
            {
                return Err(invalid(format!(
                    "Invalid native model topology at node {id}"
                )));
            }
            let len = elements(&node.output_dims)?;
            if matches!(node.operator, Operator::Input(_) | Operator::Constant(_)) {
                if !node.inputs.is_empty() {
                    return Err(invalid(format!("Unexpected inputs for source node {id}")));
                }
                let slot = graph.num_inputs();
                if let Operator::Constant(value) = &node.operator {
                    if value.0.dims() != node.output_dims || value.0.inner.len() != len {
                        return Err(invalid(format!("Invalid constant tensor at node {id}")));
                    }
                    constants.insert(slot, value.0.inner.clone());
                }
                graph.input_shapes.push(node.output_dims.clone());
                tensors.insert(id, slot);
            }
        }
        for (&id, node) in &source.nodes {
            if tensors.contains_key(&id) {
                continue;
            }
            let args = node.inputs.iter().map(|i| tensors[i]).collect::<Vec<_>>();
            let shape = node
                .inputs
                .first()
                .map(|i| source.nodes[i].output_dims.as_slice());
            let output = lower(&mut graph, node, &args, shape)
                .map_err(|e| invalid(format!("Native model node {id}: {e:?}")))?;
            tensors.insert(id, output);
        }
        graph.outputs = source
            .outputs
            .iter()
            .map(|id| {
                tensors
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("Unknown native model output"))
            })
            .collect::<Result<_, _>>()?;
        let shapes = graph.tensor_shapes()?;
        for (&id, node) in &source.nodes {
            if shapes[tensors[&id]] != node.output_dims {
                return Err(invalid(format!(
                    "Native model output shape differs at node {id}"
                )));
            }
        }
        let inputs = source
            .inputs
            .iter()
            .map(|id| tensors[id])
            .collect::<Vec<_>>();
        // Bind the external input order and original tensor map as well as the
        // native graph. Input order need not equal increasing node order.
        graph
            .context
            .extend_from_slice(&(inputs.len() as u64).to_le_bytes());
        for id in &inputs {
            graph.context.extend_from_slice(&(*id as u64).to_le_bytes());
        }
        graph
            .context
            .extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        for (id, native) in &tensors {
            graph.context.extend_from_slice(&(*id as u64).to_le_bytes());
            graph
                .context
                .extend_from_slice(&(*native as u64).to_le_bytes());
        }
        Ok(Self {
            graph,
            tensors,
            inputs,
            constants,
        })
    }

    /// The translated graph, including all auxiliary proof constraints.
    pub fn graph(&self) -> &NativeGraph {
        &self.graph
    }

    /// Locate the native tensor for an original model node, for example when
    /// constructing a boundary proof. Unknown model nodes return `None`.
    pub fn tensor_id(&self, model_node: usize) -> Option<usize> {
        self.tensors.get(&model_node).copied()
    }

    /// Native input tensor IDs in the original model's external input order.
    pub fn input_tensor_ids(&self) -> &[usize] {
        &self.inputs
    }

    /// Register public constants once. Authenticate the returned registration
    /// through the verifier's trusted setup, not through a proof bundle.
    pub fn preprocess(
        self,
        setup: &DoryProverSetup,
    ) -> Result<NativeModelPreprocessing, ProofVerifyError> {
        let inputs = self
            .inputs
            .into_iter()
            .map(|id| (id, self.graph.input_shapes[id].clone()))
            .collect();
        Ok(NativeModelPreprocessing {
            inputs,
            cache: NativeGraphPreprocessing::new(self.graph, self.constants, setup)?,
        })
    }
}

fn lower(
    graph: &mut NativeGraph,
    node: &ComputationNode,
    args: &[usize],
    shape: Option<&[usize]>,
) -> Result<usize, ProofVerifyError> {
    let arity = match &node.operator {
        Operator::Add(_) | Operator::Sub(_) | Operator::Mul(_) | Operator::Concat(_) => 2,
        Operator::Square(_)
        | Operator::Sum(_)
        | Operator::MeanOfSquares(_)
        | Operator::Rsqrt(_)
        | Operator::ScalarConstDiv(_)
        | Operator::Identity(_)
        | Operator::Broadcast(_)
        | Operator::Reshape(_)
        | Operator::MoveAxis(_)
        | Operator::Slice(_)
        | Operator::SoftmaxLastAxis(_)
        | Operator::Sin(_)
        | Operator::Cos(_) => 1,
        _ => return Err(invalid("Unsupported operator in native model frontend")),
    };
    if args.len() != arity {
        return Err(invalid("Invalid native model operator arity"));
    }
    let shape = shape.ok_or_else(|| invalid("Missing native model operand"))?;
    let input = args[0];
    let lowered = match &node.operator {
        Operator::Add(_) => NativeGraphNode::add(input, args[1]),
        Operator::Sub(_) => NativeGraphNode::sub(input, args[1]),
        Operator::Mul(op) => NativeGraphNode::mul(input, args[1], scale(op.scale)?),
        Operator::Square(op) => NativeGraphNode::mul(input, input, scale(op.scale)?),
        Operator::Sum(op) => NativeGraphNode::sum(input, op.axes.clone()),
        Operator::MeanOfSquares(op) => {
            let mut padded = 1usize;
            let mut seen = BTreeSet::new();
            for axis in &op.axes {
                if !seen.insert(*axis) {
                    return Err(invalid("Duplicate mean reduction axis"));
                }
                padded = padded
                    .checked_mul(
                        *shape
                            .get(*axis)
                            .ok_or_else(|| invalid("Invalid mean axis"))?,
                    )
                    .ok_or_else(|| invalid("Mean reduction size overflow"))?;
            }
            if op.padded_count != padded {
                return Err(invalid("Incorrect padded mean count"));
            }
            NativeGraphNode::mean_of_squares_with_count(
                input,
                op.axes.clone(),
                scale(op.scale)?,
                op.count,
            )
        }
        Operator::Rsqrt(op) => NativeGraphNode::rsqrt(input, scale(op.scale)?),
        Operator::ScalarConstDiv(op) => NativeGraphNode::div_floor(
            input,
            u32::try_from(op.divisor).map_err(|_| invalid("Unsupported native divisor"))?,
        ),
        Operator::Identity(_) => NativeGraphNode::reshape(input, shape.to_vec()),
        Operator::Broadcast(op) => {
            elements(&op.shape)?;
            NativeGraphNode::broadcast(input, op.shape.clone())
        }
        Operator::Reshape(op) => {
            elements(&op.shape)?;
            NativeGraphNode::reshape(input, op.shape.clone())
        }
        Operator::MoveAxis(op) => {
            if op.source >= shape.len() || op.destination >= shape.len() {
                return Err(invalid("Invalid move axis"));
            }
            let mut axes = (0..shape.len()).collect::<Vec<_>>();
            let axis = axes.remove(op.source);
            axes.insert(op.destination, axis);
            NativeGraphNode::permute(input, axes)
        }
        Operator::Slice(op) => NativeGraphNode::slice(
            input,
            op.axis,
            op.start,
            op.end
                .checked_sub(op.start)
                .ok_or_else(|| invalid("Invalid slice interval"))?,
        ),
        Operator::Concat(op) => {
            let rank = shape.len() as isize;
            if op.axis < -rank || op.axis >= rank {
                return Err(invalid("Invalid concat axis"));
            }
            NativeGraphNode::concat(
                input,
                args[1],
                if op.axis < 0 {
                    (rank + op.axis) as usize
                } else {
                    op.axis as usize
                },
            )
        }
        Operator::SoftmaxLastAxis(op) => {
            return Ok(inline(
                graph,
                NativeGraph::softmax_with_checked_centering(
                    b"native model softmax".to_vec(),
                    shape.to_vec(),
                    scale(op.scale)?,
                )?,
                input,
            ))
        }
        Operator::Cos(op) => {
            return Ok(inline(
                graph,
                NativeGraph::trig(
                    b"native model cosine".to_vec(),
                    shape.to_vec(),
                    scale(op.scale)?,
                    true,
                )?,
                input,
            ))
        }
        Operator::Sin(op) => {
            return Ok(inline(
                graph,
                NativeGraph::trig(
                    b"native model sine".to_vec(),
                    shape.to_vec(),
                    scale(op.scale)?,
                    false,
                )?,
                input,
            ))
        }
        _ => unreachable!("arity match rejects unsupported operators"),
    };
    Ok(append(graph, lowered))
}

/// Reusable public setup for a translated model. Private inputs are supplied in
/// `Model::graph.inputs` order and receive new hiding commitments on each call.
pub struct NativeModelPreprocessing {
    inputs: Vec<(usize, Vec<usize>)>,
    cache: NativeGraphPreprocessing,
}

impl NativeModelPreprocessing {
    /// Expected graph, public constant commitments and verifier setup. The
    /// receiver must authenticate this registration independently of the proof.
    pub fn registered(&self) -> &NativeRegisteredGraph {
        self.cache.registered()
    }

    /// Execute and commit to exact-shaped private inputs in model input order.
    /// Reuses public commitments and checks setup identity. The returned witness
    /// is private and must be passed to `NativeGraphProof::prove`, not exported.
    pub fn commit(
        &self,
        inputs: &[Tensor<i32>],
        setup: &DoryProverSetup,
    ) -> Result<(NativeGraphStatement, NativeGraphWitness), ProofVerifyError> {
        if inputs.len() != self.inputs.len() {
            return Err(invalid("Wrong native model input count"));
        }
        let mut private = BTreeMap::new();
        for ((id, shape), tensor) in self.inputs.iter().zip(inputs) {
            if tensor.dims() != shape || tensor.inner.len() != elements(shape)? {
                return Err(invalid("Wrong native model input shape or length"));
            }
            private.insert(*id, tensor.inner.clone());
        }
        self.cache.commit(private, setup)
    }
}

#[cfg(test)]
mod tests;
