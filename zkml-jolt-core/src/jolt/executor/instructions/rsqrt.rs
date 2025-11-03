use itertools::Itertools;
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{
        VirtualInstructionSequence, add::AddInstruction, beq::BeqInstruction, div::DivInstruction,
        gte::GteInstruction, mul::MulInstruction, sub::SubInstruction,
    },
    utils::{VirtualSequenceCounter, u64_vec_to_i32_iter},
};

// Scale factor
const SF: u64 = 128;
const SF_LOG: u64 = 7;
const DIV_VIRTUAL_REGISTERS: usize = 6;

/// Quantized Rsqrt producing `Q / sqrt(x)`
///
/// ### Overview
/// This implementation computes a **quantized approximation of the rsqrt function**
/// that supports unsigned inputs
/// Follows https://rusteddreams.bitbucket.io/2017/03/05/sqrt.html
///
/// The algorithm behaves as:
/// ```text
/// if z_i == 0 → z_i += 1              | clamping 0 to 1
/// if log(z_i) > 3 * log(Q) → d_i = 0  | if log(1/sqrt(x)) < 1/Q (ulp), set d = 0
/// else d_i = 3*log(Q) - log(z_i)      | log(1/sqrt(x))
/// g_i = d + d(xd^2-1)(sqrt(2)/2-1)    | Function to approximate 1/sqrt(x)
/// ```
///
/// ### Virtual trace sequence
/// The sequence performs 14 virtual steps:
///
/// | Step  | Operation             | Description                                       | Expression                    |
/// |-------|-----------------------|---------------------------------------------------|-------------------------------|
/// | 1     | `VirtualConst(Q)`     | Initialize scale_factor tensor                    | `Q`                           |
/// | 2     | `VirtualConst(1)`     | Initialize ulp tensor                             | `1`                           |
/// | 3     | `Eq`                  | Returns `x == 0`                                  | `x == 0`                      |
/// | 4     | `Select`              | Compute `x = select(x, x, ulp)`                   | `max(x, 1)`                   |
/// | 5     | `VirtualConst(a_lt_0)`| Initialize a_lt_0 tensor, if `xd^2-1 < 0`         | `2 - 2 * sqrt(2)`             |
/// | 6     | `VirtualConst(a)`     | Initialize a_gt_0 tensor, if `xd^2-1 >= 0`        | `sqrt(2)/2 - 1`               |
/// | 7     | `VirtualAdvice(d)`    | Initialize d tensor                               | `2^((3log(Q) - log(x))/2)`    |
/// | 8     | `Mul`                 | Compute `xd_ns = x * d`                           | `Q * xd`                      |
/// | 9     | `Div`                 | Compute `xd = xd_ns / Q`                          | `xd`                          |
/// | 10    | `Mul`                 | Compute `xd_sq_ns = d * xd`                       | `Q * xd^2`                    |
/// | 11    | `Div`                 | Compute `xd_sq = xd_sq_ns / Q`                    | `xd^2`                        |
/// | 12    | `Sub`                 | Compute `xd_sq_minus1 = xd_sq - 1`                | `xd^2-1`                      |
/// | 13    | `Gte`                 | Returns `gt_0 = xd^2-1 >= 0`                      | `xd^2-1 >= 0`                 |
/// | 14    | `Select`              | Compute `a = select(gt0, a_gt_0, a_lt_0)`         | `xd^2-1>=0 ? a_gt_0 : a_lt_0` |
/// | 15    | `Mul`                 | Compute `xd_cub_minusd_ns = d * xd_sq_minus1`     | `Q * d(xd^2-1)`               |
/// | 16    | `Div`                 | Compute `xd_cub_minusd = xd_cub_minusd_ns / Q`    | `d(xd^2-1)`                   |
/// | 17    | `Mul`                 | Compute `axd_cub_minusd_ns = a * xd_cub_minusd`   | `Q * d(xd^2-1)(sqrt(2)/2-1)`  |
/// | 18    | `Div`                 | Compute `axd_cub_minusd = axd_cub_minusd_ns / Q`  | `d(xd^2-1)(sqrt(2)/2-1)`      |
/// | 19    | `Add`                 | Compute `approx = d + axd_cub_minusd`             | `d + d(xd^2-1)(sqrt(2)/2-1)`  |
/// | 20    | `VirtualMove`         | Write final result to output tensor               | `d + d(xd^2-1)(sqrt(2)/2-1)`  |
///
/// Each `ONNXCycle` represents one of these steps in the virtualized trace.
///
/// ### Notes
/// - x is clamped to a minimal value of 1, since we can't divide by 0.
/// - Designed for quantized circuits or proof backends where fractional values are approximated in integer space.
pub struct RsqrtInstruction<const WORD_SIZE: usize>;
impl<const WORD_SIZE: usize> VirtualInstructionSequence for RsqrtInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 16 + 4 * DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH;

    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Rsqrt);
        let num_outputs = cycle.instr.active_output_elements;

        // If RSQRT is part of a longer virtual sequence, recover the counter to continue decrementing it
        let virtual_sequence_remaining =
            if let Some(remaining) = cycle.instr.virtual_sequence_remaining {
                assert!(
                    remaining >= Self::SEQUENCE_LENGTH,
                    "Not enough remaining virtual sequence steps"
                );
                remaining
            } else {
                Self::SEQUENCE_LENGTH
            };
        let mut vseq_counter = VirtualSequenceCounter::new(virtual_sequence_remaining);

        // RSQRT source registers
        let r_x = cycle.instr.ts1;

        // TODO(AntoineF4C5): set to 0 (requires to change `virtual_tensor_index`: generates bug on model composed of just rsqrt)
        let mut index = 1;
        // i++
        let mut ixx = || {
            let tmp = index;
            index += 1;
            tmp
        };
        // Virtual registers used in sequence
        let v_one = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_ulp = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xeq0 = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_x = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_a_lt_0 = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_a_gt_0 = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_d = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_ns = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_sq_ns = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_sq = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_sq_minus1 = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_gt0 = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_a = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_cub_minusd_ns = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_xd_cub_minusd = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_axd_cub_minusd_ns = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_axd_cub_minusd = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));
        let v_approx = Some(virtual_tensor_index(ixx(), K, cycle.instr.td.unwrap()));

        // RSQRT operand
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let mut virtual_trace = vec![];

        // const one (scaled)
        let one = vec![SF; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_one,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
            },
            advice_value: None,
        });

        // Lowest precision unit, 1/sf (quantized to 1 in zkvm)
        let ulp = vec![1; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_ulp,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&ulp))),
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&ulp))),
            },
            advice_value: None,
        });

        let xeq0 = (0..num_outputs)
            .map(|i| BeqInstruction::<WORD_SIZE>(x[i], 0).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Eq,
                ts1: r_x,
                ts2: None,
                ts3: None,
                td: v_xeq0,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; num_outputs]))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xeq0))),
            },
            advice_value: None,
        });

        // TODO(AntoineF4C5): Use clamping
        // If x == 0, we will set it to 1 (lowest non-zero value), since we can't divide by 0
        // Quantized x = 0 for any input in [0, ulp) where ulp = 1/sf
        let clamped_x: Vec<u64> = xeq0
            .iter()
            .zip(&ulp)
            .zip(&x)
            .map(|((&cond, &x), &nx)| if cond != 0 { x } else { nx })
            .collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_xeq0,
                ts2: v_ulp,
                ts3: r_x,
                td: v_x,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xeq0))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&ulp))),
                ts3_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&clamped_x))),
            },
            advice_value: None,
        });
        let x = clamped_x;

        // Constant 2 - 2 * sqrt(2), factor conditionally used to calculate approximation
        let sqrt_2 = (2f32.sqrt() * SF as f32).round() as u64;
        let a_lt_0 = (2 * SF as i32 - 2 * sqrt_2 as i32) as u32 as u64;
        let a_lt_0 = vec![a_lt_0; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_a_lt_0,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&a_lt_0))),
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_lt_0))),
            },
            advice_value: None,
        });

        // Constant sqrt(2)/2-1, factor conditionally used to calculate approximation
        let a_gt_0 = (sqrt_2 as i32 / 2 - SF as i32) as u32 as u64;
        let a_gt_0 = vec![a_gt_0; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_a_gt_0,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&a_gt_0))),
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_gt_0))),
            },
            advice_value: None,
        });

        // TODO(AntoineF4C5): Use log opcode
        // First approximation: d = log(1/sqrt(x)) = -log(x)/2
        let d = (0..num_outputs)
            .map(|i| {
                let exp = 3 * SF_LOG as i32 - x[i].ilog2() as i32;
                if exp < 0 {
                    0
                } else {
                    2_u64.pow(exp as u32 / 2)
                }
            })
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_d,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
        });

        // d*x
        let xd_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(x[i], d[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_x,
                ts2: v_d,
                ts3: None,
                td: v_xd_ns,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_ns))),
            },
            advice_value: None,
        });

        // Rescale
        let xd = DivInstruction::<WORD_SIZE>::sequence_output(
            xd_ns.clone(),
            vec![SF; num_outputs],
            None,
        );

        virtual_trace.extend(DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_xd_ns,
                    ts2: None,
                    ts3: None,
                    td: v_xd,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                    virtual_sequence_remaining: Some(vseq_counter.get()),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_ns))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd))),
                },
                advice_value: None,
            },
            K + index,
        ));
        index += DIV_VIRTUAL_REGISTERS;
        vseq_counter.substract(DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH);

        // xd^2
        let xd_sq_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(d[i], xd[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_d,
                ts2: v_xd,
                ts3: None,
                td: v_xd_sq_ns,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_ns))),
            },
            advice_value: None,
        });

        // Rescale
        let xd_sq = DivInstruction::<WORD_SIZE>::sequence_output(
            xd_sq_ns.clone(),
            vec![SF; num_outputs],
            None,
        );

        virtual_trace.extend(DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_xd_sq_ns,
                    ts2: None,
                    ts3: None,
                    td: v_xd_sq,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                    virtual_sequence_remaining: Some(vseq_counter.get()),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_ns))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq))),
                },
                advice_value: None,
            },
            K + index,
        ));
        index += DIV_VIRTUAL_REGISTERS;
        vseq_counter.substract(DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH);

        // xd^2-1
        let xd_sq_minus1 = (0..num_outputs)
            .map(|i| SubInstruction::<WORD_SIZE>(xd_sq[i], one[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_xd_sq,
                ts2: v_one,
                ts3: None,
                td: v_xd_sq_minus1,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_minus1))),
            },
            advice_value: None,
        });

        // xd^2-1 >= 0
        let gt_0: Vec<u64> = (0..num_outputs)
            .map(|i| GteInstruction::<WORD_SIZE>(xd_sq_minus1[i], 0).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gte,
                ts1: v_xd_sq_minus1,
                ts2: None,
                ts3: None,
                td: v_gt0,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_minus1))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; num_outputs]))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&gt_0))),
            },
            advice_value: None,
        });

        // a = if xd^2-1 >= 0 { sqrt(2)/2 - 1 } else { 2 - 2 * sqrt(2) }
        let a: Vec<u64> = gt_0
            .iter()
            .zip(a_gt_0.iter())
            .zip(a_lt_0.iter())
            .map(
                |((&gt0, &a_gt0), &a_lt0)| {
                    if gt0 != 0 { a_gt0 } else { a_lt0 }
                },
            )
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_gt0,
                ts2: v_a_gt_0,
                ts3: v_a_lt_0,
                td: v_a,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&gt_0))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_gt_0))),
                ts3_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_lt_0))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
            },
            advice_value: None,
        });

        // d(xd^2-1)
        let xd_cub_minusd_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(xd_sq_minus1[i], d[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_xd_sq_minus1,
                ts2: v_d,
                ts3: None,
                td: v_xd_cub_minusd_ns,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_minus1))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd_ns))),
            },
            advice_value: None,
        });

        // Rescale
        let xd_cub_minusd = DivInstruction::<WORD_SIZE>::sequence_output(
            xd_cub_minusd_ns.clone(),
            vec![SF; num_outputs],
            None,
        );
        virtual_trace.extend(DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_xd_cub_minusd_ns,
                    ts2: None,
                    ts3: None,
                    td: v_xd_cub_minusd,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                    virtual_sequence_remaining: Some(vseq_counter.get()),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd_ns))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd))),
                },
                advice_value: None,
            },
            K + index,
        ));
        index += DIV_VIRTUAL_REGISTERS;
        vseq_counter.substract(DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH);

        // ad(xd^2-1)
        let axd_cub_minusd_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(xd_cub_minusd[i], a[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_xd_cub_minusd,
                ts2: v_a,
                ts3: None,
                td: v_axd_cub_minusd_ns,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd_ns))),
            },
            advice_value: None,
        });

        // Rescale
        let axd_cub_minusd = DivInstruction::<WORD_SIZE>::sequence_output(
            axd_cub_minusd_ns.clone(),
            vec![SF; num_outputs],
            None,
        );
        virtual_trace.extend(DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_axd_cub_minusd_ns,
                    ts2: None,
                    ts3: None,
                    td: v_axd_cub_minusd,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                    virtual_sequence_remaining: Some(vseq_counter.get()),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd_ns))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd))),
                },
                advice_value: None,
            },
            K + index,
        ));
        vseq_counter.substract(DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH);

        // d + ad(xd^2-1)
        let approx: Vec<u64> = (0..num_outputs)
            .map(|i| AddInstruction::<WORD_SIZE>(d[i], axd_cub_minusd[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_d,
                ts2: v_axd_cub_minusd,
                ts3: None,
                td: v_approx,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&approx))),
            },
            advice_value: None,
        });

        // Write to td
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_approx,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&approx))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&approx))),
            },
            advice_value: None,
        });

        assert_eq!(
            virtual_trace.len(),
            Self::SEQUENCE_LENGTH,
            "Invalid virtual trace length"
        );

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, _: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        let sf_log = SF.ilog2() as i32;
        let sf = SF as i32;
        // div instruction in zkvm is flooring, (ex: -4/3 = -2). This method allows to have consistent result with zkvm.
        let rescale_down = |q: i32| {
            if q % sf < 0 { q / sf - 1 } else { q / sf }
        };

        let num_outputs = x.len();
        let mut output = vec![0; num_outputs];
        for i in 0..num_outputs {
            let sqrt_2 = (2f32.sqrt() * SF as f32).round() as i32;

            let x = if x[i] != 0 { x[i] } else { 1 };
            let d = {
                let exp = 3 * sf_log - x.ilog2() as i32;
                if exp < 0 {
                    0
                } else {
                    2_i32.pow(exp as u32 / 2)
                }
            };
            let xd = rescale_down(x as u32 as i32 * d);
            let xd_sq = rescale_down(d * xd);
            let xd_sq_minus1 = xd_sq - sf;
            let xd_cub_minusd = rescale_down(d * xd_sq_minus1);
            let a = if xd_sq_minus1 >= 0 {
                sqrt_2 / 2 - sf
            } else {
                2 * sf - 2 * sqrt_2
            };
            let axd_cub_minusd = rescale_down(a * xd_cub_minusd);
            let approximation = d + axd_cub_minusd;

            // TODO(AntoineF4C5): apply one or two rounds of Newton-Raphson method to get better results
            // Newton's method
            // let approximation = (approximation
            //     * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
            //     / (2 * sf);

            // output[i] = ((approximation
            //     * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
            //     / (2 * sf)) as u32 as u64;
            // approximation

            output[i] = approximation as u32 as u64;
        }
        output
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;

    use super::*;
    use num_traits::Pow;
    use rand::prelude::*;

    #[test]
    fn rsqrt_virtual_sequence_32() {
        jolt_virtual_sequence_test::<RsqrtInstruction<32>>(ONNXOpcode::Rsqrt, 1);
    }

    #[test]
    fn rsqrt_precision() {
        let mut rng = StdRng::seed_from_u64(123456);
        let sf = SF as f32;
        let nb_loops = 100_000;

        let mut valid_tries = 0;
        let mut total_error = 0.0;

        let mut total_offset = 0;
        for _ in 0..nb_loops {
            let input: f32 = (rng.r#gen::<f32>()) * 2f32.pow(10); // Select values between 0 and 1000 (quantized to [0..128_000])
            let expected = 1.0 / input.sqrt();

            // Input to the zkvm, this is quantized with a precision of 1/sf.
            let x = (input * sf) as i32 as u32 as u64; // cast to vm compatible type
            let result = RsqrtInstruction::<32>::sequence_output(vec![x], vec![], None)[0];
            let dequantized = result as f32 / sf;
            let error = (dequantized - expected).abs() / expected;
            // If result == 0, we consider the offset between quantized expected result and 0.
            if result != 0 {
                valid_tries += 1;
                total_error += error;
            } else {
                let offset = (expected * sf) as u32;
                total_offset += offset;
            }
        }
        println!(
            "mean error(if res!=0): {:4.2}%",
            total_error * 100.0 / valid_tries as f32
        );
        println!(
            "res=0:    {}%",
            100.0 * (1.0 - valid_tries as f32 / nb_loops as f32)
        );
        println!(
            "avg offset(if res=0):  {:} ulp",
            total_offset as f32 / (nb_loops - valid_tries) as f32
        );
    }
}
