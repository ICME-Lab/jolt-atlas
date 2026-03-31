//! f64 shadow execution for quantization error analysis.
//!
//! Runs the quantized computation graph in lockstep with an f64 "shadow"
//! that performs the mathematically ideal version of each operation. After
//! each node, computes drift metrics (cosine similarity, relative MSE, etc.)
//! between the dequantized i32 output and the f64 reference.

use crate::{
    model::Model,
    ops::{Op, Operator},
    tensor::{self, Tensor, TensorType},
    utils::{
        metrics,
        quantize::{self, Scale},
    },
};
use std::collections::BTreeMap;
use tracing::debug;

/// Map from decomposed graph node index → original f64 tensor (before quantization).
pub type OriginalF64Constants = BTreeMap<usize, Tensor<f64>>;

// ────────────────────────────────────────────────────────────────────────────
// Per-node metrics
// ────────────────────────────────────────────────────────────────────────────

/// Metrics describing the drift at a single computation node.
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    /// Node index in the computation graph.
    pub idx: usize,
    /// Short operator name (e.g. "Add", "Einsum", "SoftmaxAxes").
    pub op_name: String,
    /// Output tensor shape.
    pub dims: Vec<usize>,
    /// Number of output elements.
    pub numel: usize,
    /// Cosine similarity between dequantized i32 output and f64 shadow output.
    pub cosine_sim: f64,
    /// Relative MSE: MSE(deq, f64) / Var(f64).
    pub relative_mse: f64,
    /// Maximum absolute error: max |deq_i - f64_i|.
    pub max_abs_err: f64,
    /// Mean absolute error.
    pub mean_abs_err: f64,
    /// RMS of dequantized i32 output.
    pub deq_rms: f64,
    /// RMS of f64 shadow output.
    pub ref_rms: f64,
}

impl NodeMetrics {
    /// Trivially perfect metrics for an input node.
    fn input(idx: usize, tensor: &Tensor<f64>) -> Self {
        Self {
            idx,
            op_name: "Input".to_string(),
            dims: tensor.dims().to_vec(),
            numel: tensor.len(),
            cosine_sim: 1.0,
            relative_mse: 0.0,
            max_abs_err: 0.0,
            mean_abs_err: 0.0,
            deq_rms: 0.0,
            ref_rms: 0.0,
        }
    }

    /// Placeholder metrics for a node in the true-f64 shadow (no i32 comparison).
    fn passthrough(idx: usize, op_name: String, tensor: &Tensor<f64>) -> Self {
        Self {
            idx,
            op_name,
            dims: tensor.dims().to_vec(),
            numel: tensor.len(),
            cosine_sim: 1.0,
            relative_mse: 0.0,
            max_abs_err: 0.0,
            mean_abs_err: 0.0,
            deq_rms: 0.0,
            ref_rms: 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ShadowTrace result
// ────────────────────────────────────────────────────────────────────────────

/// Result of a shadow-traced execution.
pub struct ShadowTrace {
    /// Per-node metrics, in graph execution order.
    pub node_metrics: Vec<NodeMetrics>,
    /// The f64 shadow outputs (all nodes).
    pub f64_outputs: BTreeMap<usize, Tensor<f64>>,
    /// The i32 quantized outputs (all nodes).
    pub i32_outputs: BTreeMap<usize, Tensor<i32>>,
}

impl ShadowTrace {
    /// Print a formatted per-node drift report.
    pub fn print_report(&self) {
        debug!(
            "{:<6} {:<18} {:<20} {:<8} {:<10} {:<12} {:<12} {:<12} {:<12} {:<12}",
            "Node",
            "Op",
            "Shape",
            "Numel",
            "CosSim",
            "RelMSE",
            "MaxAbsErr",
            "MeanAbsErr",
            "DeqRMS",
            "RefRMS"
        );
        debug!("{}", "-".repeat(130));
        for m in &self.node_metrics {
            debug!(
                "{:<6} {:<18} {:<20} {:<8} {:<10} {:<12} {:<12.6} {:<12.6} {:<12.6} {:<12.6}",
                m.idx,
                m.op_name,
                format!("{:?}", m.dims),
                m.numel,
                format_metric(m.cosine_sim),
                format_metric(m.relative_mse),
                m.max_abs_err,
                m.mean_abs_err,
                m.deq_rms,
                m.ref_rms,
            );
        }
    }

    /// Aggregate metrics by operator class and print summary.
    pub fn print_op_class_summary(&self) {
        use std::collections::BTreeMap as Map;
        struct Agg {
            count: usize,
            cos_sum: f64,
            cos_worst: f64,
            rmse_sum: f64,
            rmse_worst: f64,
            max_abs_worst: f64,
        }

        let mut agg: Map<String, Agg> = Map::new();
        for m in &self.node_metrics {
            let e = agg.entry(m.op_name.clone()).or_insert(Agg {
                count: 0,
                cos_sum: 0.0,
                cos_worst: 1.0,
                rmse_sum: 0.0,
                rmse_worst: 0.0,
                max_abs_worst: 0.0,
            });
            e.count += 1;
            if !m.cosine_sim.is_nan() {
                e.cos_sum += m.cosine_sim;
                e.cos_worst = e.cos_worst.min(m.cosine_sim);
            }
            let rmse = if m.relative_mse.is_finite() {
                m.relative_mse
            } else {
                0.0
            };
            e.rmse_sum += rmse;
            e.rmse_worst = e.rmse_worst.max(rmse);
            e.max_abs_worst = e.max_abs_worst.max(m.max_abs_err);
        }

        debug!(
            "{:<18} {:<6} {:<12} {:<12} {:<12} {:<12} {:<12}",
            "Op Class",
            "Count",
            "Avg CosSim",
            "Worst CosSim",
            "Avg RelMSE",
            "Worst RelMSE",
            "Worst MaxAbs"
        );
        debug!("{}", "-".repeat(84));
        for (op, a) in &agg {
            debug!(
                "{:<18} {:<6} {:<12.6} {:<12.6} {:<12.6} {:<12.6} {:<12.6}",
                op,
                a.count,
                a.cos_sum / a.count as f64,
                a.cos_worst,
                a.rmse_sum / a.count as f64,
                a.rmse_worst,
                a.max_abs_worst,
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Shadow execution
// ────────────────────────────────────────────────────────────────────────────

impl Model {
    /// Execute the graph with both i32 (quantized) and f64 (shadow) paths,
    /// computing per-node drift metrics.
    ///
    /// # Arguments
    ///
    /// * `i32_inputs` — Quantized i32 input tensors (same as `forward()`).
    /// * `f64_inputs` — Unquantized f64 input tensors (original real values).
    /// * `scale`      — The quantization scale used.
    pub fn trace_with_shadow(
        &self,
        i32_inputs: &[Tensor<i32>],
        f64_inputs: &[Tensor<f64>],
        scale: Scale,
    ) -> ShadowTrace {
        let mut i32_outputs: BTreeMap<usize, Tensor<i32>> = BTreeMap::new();
        let mut f64_outputs: BTreeMap<usize, Tensor<f64>> = BTreeMap::new();

        // Store inputs
        self.store_i32_inputs(i32_inputs, &mut i32_outputs);
        self.store_f64_inputs(f64_inputs, &mut f64_outputs);

        let mut node_metrics = Vec::new();

        for (node_idx, node) in &self.graph.nodes {
            if matches!(node.operator, Operator::Input(_)) {
                if let Some(f64_t) = f64_outputs.get(node_idx) {
                    node_metrics.push(NodeMetrics::input(*node_idx, f64_t));
                }
                continue;
            }

            // ── i32 path (mirrors execute_graph rescaling logic) ────────
            let i32_input_tensors: Vec<&Tensor<i32>> = node
                .inputs
                .iter()
                .map(|&idx| i32_outputs.get(&idx).unwrap())
                .collect();
            let i32_out = node.operator.f(i32_input_tensors);

            // ── f64 shadow path ─────────────────────────────────────────
            let f64_input_tensors: Vec<&Tensor<f64>> = node
                .inputs
                .iter()
                .map(|&idx| f64_outputs.get(&idx).unwrap())
                .collect();
            let f64_out = shadow_f64(&node.operator, f64_input_tensors, scale);

            // ── Compute metrics ─────────────────────────────────────────
            let op_name = op_variant_name(&node.operator);
            let deq = dequantize_tensor(&i32_out, scale);
            node_metrics.push(compute_metrics(*node_idx, &op_name, &deq, &f64_out));
            i32_outputs.insert(*node_idx, i32_out);
            f64_outputs.insert(*node_idx, f64_out);
        }

        ShadowTrace {
            node_metrics,
            f64_outputs,
            i32_outputs,
        }
    }

    /// Store i32 inputs into the node output map (mirrors execute.rs logic).
    fn store_i32_inputs(&self, inputs: &[Tensor<i32>], outputs: &mut BTreeMap<usize, Tensor<i32>>) {
        self.store_shadow_inputs(inputs, outputs);
    }

    /// Store f64 inputs into the node output map.
    fn store_f64_inputs(&self, inputs: &[Tensor<f64>], outputs: &mut BTreeMap<usize, Tensor<f64>>) {
        self.store_shadow_inputs(inputs, outputs);
    }

    /// Generic input storage — pads to power-of-2 dims when necessary.
    fn store_shadow_inputs<T: Clone + TensorType + Send + Sync>(
        &self,
        inputs: &[Tensor<T>],
        outputs: &mut BTreeMap<usize, Tensor<T>>,
    ) {
        for (i, tensor) in inputs.iter().enumerate() {
            let idx = self.graph.inputs[i];
            if let Some(original_dims) = self.graph.original_input_dims.get(&idx) {
                assert_eq!(tensor.dims(), original_dims.as_slice());
                let node = self.graph.nodes.get(&idx).unwrap();
                let mut padded = tensor.clone();
                padded.pad_to_dims(&node.output_dims).expect("pad failed");
                outputs.insert(idx, padded);
            } else {
                outputs.insert(idx, tensor.clone());
            }
        }
    }

    /// Load the original f32 constants from the ONNX file via Tract, convert
    /// to f64, and map them to the decomposed graph's Constant node indices.
    ///
    /// This allows creating a "true f64 shadow" that uses the original weight
    /// values instead of quantized-then-dequantized values.
    ///
    /// The mapping works because: GraphParser visits Tract nodes in order, and
    /// each Const node produces exactly one decomposed Constant node. Both
    /// iterations are ordered, so the k-th Tract Const maps to the k-th
    /// decomposed Constant.
    pub fn load_original_f64_constants(
        &self,
        path: &str,
        run_args: &crate::model::RunArgs,
    ) -> OriginalF64Constants {
        use tract_onnx::tract_hir::ops::konst::Const;

        let (typed_model, _symbol_values) = Self::load_onnx_using_tract(path, run_args);

        // Collect Tract Const tensors in graph node order → our Tensor<f64>
        let mut tract_f64_consts: Vec<Tensor<f64>> = Vec::new();
        for node in typed_model.nodes.iter() {
            if let Some(const_op) = node.op().downcast_ref::<Const>() {
                let f32_tensor: Tensor<f32> =
                    crate::utils::parser::extract_tensor_value(const_op.val().clone())
                        .expect("failed to extract f32 constant from Tract");
                let f64_tensor: Tensor<f64> = f32_tensor.map(|v| v as f64);
                tract_f64_consts.push(f64_tensor);
            }
        }

        // Collect decomposed Constant node indices in graph order
        let const_node_indices: Vec<usize> = self
            .graph
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.operator, Operator::Constant(_)))
            .map(|(&idx, _)| idx)
            .collect();

        assert_eq!(
            tract_f64_consts.len(),
            const_node_indices.len(),
            "Tract has {} Const nodes but decomposed graph has {} Constant nodes",
            tract_f64_consts.len(),
            const_node_indices.len(),
        );

        let mut map = BTreeMap::new();
        for (tract_const, &graph_idx) in tract_f64_consts.into_iter().zip(const_node_indices.iter())
        {
            // The decomposed constant may be padded to power-of-2 dims.
            // Pad the original f64 constant to match.
            let graph_node = &self.graph.nodes[&graph_idx];
            let target_dims = &graph_node.output_dims;
            let mut padded: Tensor<f64> = tract_const;
            if padded.dims() != target_dims.as_slice() {
                padded
                    .pad_to_dims(target_dims)
                    .expect("failed to pad original constant");
            }
            map.insert(graph_idx, padded);
        }
        map
    }

    /// Execute the graph with a "true f64" shadow that uses original f32
    /// constants (from Tract) instead of dequantized i32 constants.
    ///
    /// This isolates the effect of **constant quantization** vs **computation
    /// rounding** by running the exact same decomposed graph with full-precision
    /// weights.
    pub fn trace_with_true_f64_shadow(
        &self,
        f64_inputs: &[Tensor<f64>],
        original_constants: &OriginalF64Constants,
        scale: Scale,
    ) -> ShadowTrace {
        let mut f64_outputs: BTreeMap<usize, Tensor<f64>> = BTreeMap::new();
        // No i32 path — just run f64
        self.store_f64_inputs(f64_inputs, &mut f64_outputs);

        let mut node_metrics = Vec::new();

        for (node_idx, node) in &self.graph.nodes {
            if matches!(node.operator, Operator::Input(_)) {
                if let Some(f64_t) = f64_outputs.get(node_idx) {
                    node_metrics.push(NodeMetrics::input(*node_idx, f64_t));
                }
                continue;
            }

            let f64_input_tensors: Vec<&Tensor<f64>> = node
                .inputs
                .iter()
                .map(|&idx| f64_outputs.get(&idx).unwrap())
                .collect();

            let f64_out = shadow_f64_with_originals(
                &node.operator,
                f64_input_tensors,
                scale,
                *node_idx,
                original_constants,
            );

            let op_name = op_variant_name(&node.operator);
            node_metrics.push(NodeMetrics::passthrough(*node_idx, op_name, &f64_out));
            f64_outputs.insert(*node_idx, f64_out);
        }

        ShadowTrace {
            node_metrics,
            f64_outputs,
            i32_outputs: BTreeMap::new(), // empty — no i32 path in this mode
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Shadow f64 dispatch — runs the ideal f64 version of each operator
// ────────────────────────────────────────────────────────────────────────────

fn shadow_f64(op: &Operator, inputs: Vec<&Tensor<f64>>, scale: Scale) -> Tensor<f64> {
    match op {
        // ── Arithmetic (linear, exact in f64) ───────────────────────────
        Operator::Add(_) => tensor::ops::add(&inputs).unwrap(),
        Operator::Sub(_) => tensor::ops::sub(&inputs).unwrap(),
        Operator::Neg(_) => tensor::ops::neg(inputs[0]).unwrap(),
        Operator::Mul(_) => tensor::ops::mult(&inputs).unwrap(),
        Operator::Div(_) => {
            elementwise_f64_binary(
                inputs[0],
                inputs[1],
                |x, y| {
                    if y != 0.0 { x / y } else { 0.0 }
                },
            )
        }

        // ── ScalarConstDiv ──────────────────────────────────────────────
        // Heuristic: if divisor is a power of 2 and equals 2^(k*scale) for
        // small k, treat as rebase → identity. Otherwise, real division.
        Operator::ScalarConstDiv(scd) => {
            if is_rebase_divisor(scd.divisor, scale) {
                inputs[0].clone()
            } else {
                let d = scd.divisor as f64;
                elementwise_f64(inputs[0], |x| x / d)
            }
        }

        // ── Matrix / tensor contraction ─────────────────────────────────
        Operator::Einsum(e) => tensor::ops::einsum(&e.equation, &inputs).unwrap(),

        // ── Nonlinearities (ideal f64 versions) ─────────────────────────
        Operator::SoftmaxAxes(s) => softmax_f64(inputs[0], s.axes),
        Operator::SoftmaxLastAxis { .. } => softmax_f64(inputs[0], inputs[0].dims().len() - 1),
        Operator::Clamp(_) => inputs[0].clone(), // quantization artifact — identity in f64
        Operator::Erf(_) => elementwise_f64(inputs[0], erf_f64),
        Operator::Tanh(_) => elementwise_f64(inputs[0], |x| (x).tanh()),
        Operator::Rsqrt(_) => {
            elementwise_f64(inputs[0], |x| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 })
        }
        Operator::Cos(_) => elementwise_f64(inputs[0], f64::cos),
        Operator::Sin(_) => elementwise_f64(inputs[0], f64::sin),
        Operator::ReLU(_) => elementwise_f64(inputs[0], |x| x.max(0.0)),

        // ── Reduction ───────────────────────────────────────────────────
        Operator::Sum(s) => tensor::ops::sum_axes(inputs[0], &s.axes).unwrap(),

        // ── Fused reduction ──────────────────────────────────────────────
        Operator::MeanOfSquares(m) => {
            let squared = inputs[0].pow(2).unwrap();
            let summed = tensor::ops::sum_axes(&squared, &m.axes).unwrap();
            let count: f64 = m
                .axes
                .iter()
                .map(|&ax| inputs[0].dims()[ax] as f64)
                .product();
            elementwise_f64(&summed, |x| x / count)
        }

        // ── Power ───────────────────────────────────────────────────────
        Operator::Square(_) => inputs[0].pow(2).unwrap(),
        Operator::Cube(_) => inputs[0].pow(3).unwrap(),

        // ── Index / Lookup ──────────────────────────────────────────────
        Operator::Gather(g) => {
            let indices: Tensor<usize> = inputs[1].map(|v| v as usize);
            tensor::ops::gather(inputs[0], &indices, g.axis).unwrap()
        }

        // ── Shape manipulation ──────────────────────────────────────────
        Operator::Broadcast(b) => inputs[0].expand(&b.shape).unwrap(),
        Operator::Reshape(r) => {
            let mut t = inputs[0].clone();
            t.reshape(&r.shape).unwrap();
            t
        }
        Operator::MoveAxis(m) => inputs[0]
            .clone()
            .move_axis(m.source, m.destination)
            .unwrap(),
        Operator::Identity(_) => inputs[0].clone(),
        Operator::Concat(c) => {
            assert!(!inputs.is_empty(), "Concat requires at least one input");
            let rank = inputs[0].dims().len() as isize;
            assert!(
                (-rank..rank).contains(&c.axis),
                "Axis out of bounds for Concat"
            );
            let axis = if c.axis < 0 {
                (rank + c.axis) as usize
            } else {
                c.axis as usize
            };
            tensor::ops::concat(&inputs, axis).unwrap()
        }

        // ── Constant ────────────────────────────────────────────────────
        Operator::Constant(c) => dequantize_tensor(&c.0, scale),

        // ── Logic / masking ─────────────────────────────────────────────
        Operator::Iff(_) => {
            // Treat any nonzero mask value as true.
            let (mask, a, b) = (inputs[0], inputs[1], inputs[2]);
            let out: Vec<f64> = mask
                .inner
                .iter()
                .zip(a.inner.iter().zip(b.inner.iter()))
                .map(|(m, (av, bv))| if m.abs() > 1e-12 { *av } else { *bv })
                .collect();
            Tensor::new(Some(&out), mask.dims()).unwrap()
        }
        Operator::And(_) => {
            let (a, b) = (inputs[0], inputs[1]);
            let out: Vec<f64> = a
                .inner
                .iter()
                .zip(b.inner.iter())
                .map(|(av, bv)| {
                    if av.abs() > 1e-12 && bv.abs() > 1e-12 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            Tensor::new(Some(&out), a.dims()).unwrap()
        }
        Operator::IsNan(op) => Tensor::new(
            Some(&vec![0.0_f64; op.out_dims.iter().product()]),
            &op.out_dims,
        )
        .unwrap(),

        // ── Input (should not be reached) ───────────────────────────────
        Operator::Input(_) => panic!("Input nodes should be handled separately"),
    }
}

/// Like `shadow_f64`, but for Constant nodes uses the original f64 values
/// instead of dequantizing quantized i32 constants.
fn shadow_f64_with_originals(
    op: &Operator,
    inputs: Vec<&Tensor<f64>>,
    scale: Scale,
    node_idx: usize,
    original_constants: &OriginalF64Constants,
) -> Tensor<f64> {
    if let Operator::Constant(_) = op {
        if let Some(original) = original_constants.get(&node_idx) {
            return original.clone();
        }
    }
    // Fall back to standard shadow for all other ops (and unmapped constants)
    shadow_f64(op, inputs, scale)
}

/// Apply an element-wise f64 operation, returning a new tensor with the same shape.
fn elementwise_f64(t: &Tensor<f64>, f: impl Fn(f64) -> f64) -> Tensor<f64> {
    let data: Vec<f64> = t.data().iter().map(|&x| f(x)).collect();
    Tensor::new(Some(&data), t.dims()).unwrap()
}

/// Apply an element-wise binary f64 operation.
fn elementwise_f64_binary(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    f: impl Fn(f64, f64) -> f64,
) -> Tensor<f64> {
    let data: Vec<f64> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Tensor::new(Some(&data), a.dims()).unwrap()
}

/// Format a metric value, handling NaN and Inf.
fn format_metric(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        "Inf".to_string()
    } else {
        format!("{v:.6}")
    }
}

/// f64 softmax along a given axis.
fn softmax_f64(input: &Tensor<f64>, axis: usize) -> Tensor<f64> {
    let dims = input.dims().to_vec();
    let mut output = input.clone();
    let data = output.inner.as_mut_slice();

    // Compute strides
    let axis_size = dims[axis];
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // Compute exp and sum
            let mut sum = 0.0_f64;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                data[idx] = (data[idx] - max_val).exp();
                sum += data[idx];
            }
            // Normalize
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                data[idx] /= sum;
            }
        }
    }
    output
}

/// Check if a divisor is a rebase divisor (power of `2^(k * scale)` for k=1 or k=2).
fn is_rebase_divisor(divisor: i32, scale: Scale) -> bool {
    if divisor <= 0 {
        return false;
    }
    // Check if divisor is a power of 2
    if divisor & (divisor - 1) != 0 {
        return false;
    }
    // divisor = 2^n, check if n is a multiple of scale
    let n = divisor.trailing_zeros() as i32;
    n > 0 && n % scale == 0
}

/// Dequantize an i32 tensor to f64 using the given scale.
fn dequantize_tensor(t: &Tensor<i32>, scale: Scale) -> Tensor<f64> {
    let data: Vec<f64> = t
        .data()
        .iter()
        .map(|&v| quantize::dequantize(v, scale))
        .collect();
    Tensor::construct(data, t.dims().to_vec())
}

/// Extract a short operator name from the Operator enum.
fn op_variant_name(op: &Operator) -> String {
    let dbg = format!("{op:?}");
    dbg.split('(').next().unwrap_or(&dbg).to_string()
}

/// Compute drift metrics between a dequantized tensor and the f64 reference.
fn compute_metrics(
    idx: usize,
    op_name: &str,
    dequantized: &Tensor<f64>,
    reference: &Tensor<f64>,
) -> NodeMetrics {
    let deq_data = dequantized.data();
    let ref_data = reference.data();
    let dims = reference.dims().to_vec();
    let numel = ref_data.len();

    // If shapes differ (shouldn't happen, but safety), report NaN
    if deq_data.len() != ref_data.len() {
        return NodeMetrics {
            idx,
            op_name: op_name.to_string(),
            dims,
            numel,
            cosine_sim: f64::NAN,
            relative_mse: f64::NAN,
            max_abs_err: f64::NAN,
            mean_abs_err: f64::NAN,
            deq_rms: 0.0,
            ref_rms: 0.0,
        };
    }

    if numel == 0 {
        return NodeMetrics {
            idx,
            op_name: op_name.to_string(),
            dims,
            numel: 0,
            cosine_sim: 1.0,
            relative_mse: 0.0,
            max_abs_err: 0.0,
            mean_abs_err: 0.0,
            deq_rms: 0.0,
            ref_rms: 0.0,
        };
    }

    let deq_rms = (deq_data.iter().map(|x| x * x).sum::<f64>() / numel as f64).sqrt();
    let ref_rms = (ref_data.iter().map(|x| x * x).sum::<f64>() / numel as f64).sqrt();

    NodeMetrics {
        idx,
        op_name: op_name.to_string(),
        dims,
        numel,
        cosine_sim: metrics::cosine_similarity(deq_data, ref_data),
        relative_mse: metrics::relative_mse(ref_data, deq_data),
        max_abs_err: metrics::max_abs_error(deq_data, ref_data),
        mean_abs_err: metrics::mean_abs_error(deq_data, ref_data),
        deq_rms,
        ref_rms,
    }
}

/// Compute the error function erf(x) using an Abramowitz & Stegun approximation.
///
/// Maximum error: ~1.5e-7, which is far more precise than our quantization.
fn erf_f64(x: f64) -> f64 {
    // Abramowitz & Stegun formula 7.1.26
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
