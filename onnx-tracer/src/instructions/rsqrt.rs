use crate::{
    instructions::{
        add::Add, declare_onnx_instr, eq::Eq, gte::Gte, mul::Mul, sra::Sra, sub::Sub, ElementWise,
        VirtualInstructionSequence, WORD_SIZE,
    },
    tensor::Tensor,
    trace_types::{
        AtlasCycle, AtlasInstr, AtlasOpcode, MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode,
    },
    utils::{u64_vec_to_i32_iter, VirtualSequenceCounter, VirtualSlotCounter},
};

// Scale factor
const SF: i32 = 128;
const SF_LOG: i32 = 7;

// Expandable
declare_onnx_instr!(name = Rsqrt);

impl ElementWise for Rsqrt {
    fn exec(x: u64, _y: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let sqrt_2 = (2f32.sqrt() * SF as f32).round() as i32;

                let x = if x != 0 { x } else { 1 };
                let d = {
                    let exp = 3 * SF_LOG - x.ilog2() as i32;
                    if exp < 0 {
                        0
                    } else {
                        2_i32.pow(exp as u32 / 2)
                    }
                };
                let xd = rescale_down(x as i32 * d);
                let xd_sq = rescale_down(d * xd);
                let xd_sq_minus1 = xd_sq - SF;
                let xd_cub_minusd = rescale_down(d * xd_sq_minus1);
                let a = if xd_sq_minus1 >= 0 {
                    sqrt_2 / 2 - SF
                } else {
                    2 * SF - 2 * sqrt_2
                };
                let axd_cub_minusd = rescale_down(a * xd_cub_minusd);
                (d + axd_cub_minusd) as u32 as u64

                // TODO(AntoineF4C5): apply one or two rounds of Newton-Raphson method to get better results
                // Newton's method
                // let approximation = (approximation
                //     * (3 * sf - ((approximation * q_x) >> sf_log * approximation) >> sf_log))
                //     / (2 * sf);

                // let approximation = ((approximation
                //     * (3 * sf - ((approximation * q_x) >> sf_log * approximation) >> sf_log))
                //     / (2 * sf)) as u32 as u64;

                // approximation
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

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
/// if xd^2-1>=0 → a_i = sqrt(2)/2-1    | conditionally set a to the most appropriate value
/// else a_i = 2 - 2 * sqrt(2)          | depending on whether xd^2 >= 1
/// g_i = d + ad(xd^2-1)                | Function to approximate 1/sqrt(x)
/// ```
///
/// ### Virtual trace sequence
/// The sequence performs 14 virtual steps:
///
/// | Step  | Operation             | Description                                           | Expression                    |
/// |-------|-----------------------|-------------------------------------------------------|-------------------------------|
/// | 1     | `Constant(sf)`        | Initialize scale_factor tensor                        | `sf`                          |
/// | 2     | `Constant(1)`         | Initialize ulp tensor                                 | `1`                           |
/// | 3     | `Constant(sf_log)`    | Initialize scale_factor logarithm tensor              | `sf_log`                      |
/// | 4     | `Eq`                  | Returns `x == 0`                                      | `x == 0`                      |
/// | 5     | `Select`              | Compute `x = select(x, x, ulp)`                       | `max(x, 1)`                   |
/// | 6     | `Constant(a_lt_0)`    | Initialize a_lt_0 tensor, if `xd^2-1 < 0`             | `2 - 2 * sqrt(2)`             |
/// | 7     | `Constant(a_gt_0)`    | Initialize a_gt_0 tensor, if `xd^2-1 >= 0`            | `sqrt(2)/2 - 1`               |
/// | 8     | `VirtualAdvice(d)`    | Initialize d tensor                                   | `2^((3log(sf) - log(x))/2)`   |
/// | 9     | `Mul`                 | Compute `xd_ns = x * d`                               | `sf * xd`                     |
/// | 10    | `Sra`                 | Compute `xd = xd_ns >> sf_log`                        | `xd`                          |
/// | 11    | `Mul`                 | Compute `xd_sq_ns = d * xd`                           | `sf * xd^2`                   |
/// | 12    | `Sra`                 | Compute `xd_sq = xd_sq_ns >> sf_log`                  | `xd^2`                        |
/// | 13    | `Sub`                 | Compute `xd_sq_minus1 = xd_sq - 1`                    | `xd^2-1`                      |
/// | 14    | `Gte`                 | Returns `gt_0 = xd^2-1 >= 0`                          | `xd^2-1 >= 0`                 |
/// | 15    | `Select`              | Compute `a = select(gt0, a_gt_0, a_lt_0)`             | `xd^2-1>=0 ? a_gt_0 : a_lt_0` |
/// | 16    | `Mul`                 | Compute `xd_cub_minusd_ns = d * xd_sq_minus1`         | `sf * d(xd^2-1)`              |
/// | 17    | `Sra`                 | Compute `xd_cub_minusd = xd_cub_minusd_ns >> sf_log`  | `d(xd^2-1)`                   |
/// | 18    | `Mul`                 | Compute `axd_cub_minusd_ns = a * xd_cub_minusd`       | `sf * d(xd^2-1)(sqrt(2)/2-1)` |
/// | 19    | `Sra`                 | Compute `axd_cub_minusd = axd_cub_minusd_ns >> sf_log`| `d(xd^2-1)(sqrt(2)/2-1)`      |
/// | 20    | `Add`                 | Compute `approx = d + axd_cub_minusd`                 | `d + d(xd^2-1)(sqrt(2)/2-1)`  |
/// | 21    | `VirtualMove`         | Write final result to output tensor                   | `d + d(xd^2-1)(sqrt(2)/2-1)`  |
///
/// Each `ONNXCycle` represents one of these steps in the virtualized trace.
///
/// ### Notes
/// - x is clamped to a minimal value of 1, since we can't divide by 0.
/// - Designed for quantized circuits or proof backends where fractional values are approximated in integer space.
impl VirtualInstructionSequence for Rsqrt {
    const SEQUENCE_LENGTH: usize = 17 + 4 * Sra::SEQUENCE_LENGTH;

    fn virtual_trace(cycle: ONNXCycle, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Rsqrt);
        let num_outputs = cycle.instr.num_output_elements();

        // If RSQRT is part of a longer virtual sequence, recover the counter to continue decrementing it
        let remaining = cycle
            .instr
            .virtual_sequence_remaining
            .unwrap_or(Self::SEQUENCE_LENGTH);
        assert!(
            remaining >= Self::SEQUENCE_LENGTH,
            "Not enough remaining virtual sequence steps"
        );
        let mut counter = VirtualSequenceCounter::new(remaining);

        // RSQRT source registers
        let r_x = cycle.instr.ts1;
        let td = cycle.instr.td;

        // Virtual registers used in sequence
        let v_one = Some(virtual_slot.inc());
        let v_ulp = Some(virtual_slot.inc());
        let v_sf_log = Some(virtual_slot.inc());
        let v_xeq0 = Some(virtual_slot.inc());
        let v_x = Some(virtual_slot.inc());
        let v_a_lt_0 = Some(virtual_slot.inc());
        let v_a_gt_0 = Some(virtual_slot.inc());
        let v_d = Some(virtual_slot.inc());
        let v_xd_ns = Some(virtual_slot.inc());
        let v_xd = Some(virtual_slot.inc());
        let v_xd_sq_ns = Some(virtual_slot.inc());
        let v_xd_sq = Some(virtual_slot.inc());
        let v_xd_sq_minus1 = Some(virtual_slot.inc());
        let v_gt0 = Some(virtual_slot.inc());
        let v_a = Some(virtual_slot.inc());
        let v_xd_cub_minusd_ns = Some(virtual_slot.inc());
        let v_xd_cub_minusd = Some(virtual_slot.inc());
        let v_axd_cub_minusd_ns = Some(virtual_slot.inc());
        let v_axd_cub_minusd = Some(virtual_slot.inc());
        let v_approx = Some(virtual_slot.inc());

        // RSQRT operand
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let mut virtual_trace = vec![];

        // const one (scaled)
        let one = vec![SF as u64; num_outputs];
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_one,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
                virtual_sequence_remaining: Some(counter.dec()),
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
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_ulp,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&ulp))),
                virtual_sequence_remaining: Some(counter.dec()),
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

        // scale factor logarithm, used to rescale down
        let sf_log = vec![SF_LOG as u64; num_outputs];
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_sf_log,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
            },
            advice_value: None,
        });

        let xeq0 = Eq::sequence_output(&x, &vec![0; num_outputs]);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Eq,
                ts1: r_x,
                ts2: None,
                ts3: None,
                td: v_xeq0,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Select,
                ts1: v_xeq0,
                ts2: v_ulp,
                ts3: r_x,
                td: v_x,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let sqrt_2 = (2f32.sqrt() * SF as f32).round() as u32 as i32;
        let a_lt_0 = (2 * SF - 2 * sqrt_2) as u32 as u64;
        let a_lt_0 = vec![a_lt_0; num_outputs];
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_a_lt_0,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&a_lt_0))),
                virtual_sequence_remaining: Some(counter.dec()),
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
        let a_gt_0 = (sqrt_2 / 2 - SF) as u32 as u64;
        let a_gt_0 = vec![a_gt_0; num_outputs];
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_a_gt_0,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&a_gt_0))),
                virtual_sequence_remaining: Some(counter.dec()),
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
        let d: Vec<u64> = (0..num_outputs)
            .map(|i| {
                let exp = 3 * SF_LOG - x[i].ilog2() as i32;
                if exp < 0 {
                    0
                } else {
                    2_u64.pow(exp as u32 / 2)
                }
            })
            .collect();
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_d,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let xd_ns = Mul::sequence_output(&x, &d);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Mul,
                ts1: v_x,
                ts2: v_d,
                ts3: None,
                td: v_xd_ns,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let xd = Sra::sequence_output(&xd_ns, &vec![SF_LOG as u64; num_outputs]);
        virtual_trace.extend(Sra::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sra,
                    ts1: v_xd_ns,
                    ts2: v_sf_log,
                    ts3: None,
                    td: v_xd,
                    imm: None,
                    virtual_sequence_remaining: Some(counter.get()),
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_ns))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd))),
                },
                advice_value: None,
            },
            virtual_slot,
        ));
        counter.subtract(Sra::SEQUENCE_LENGTH);

        // xd^2
        let xd_sq_ns = Mul::sequence_output(&d, &xd);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Mul,
                ts1: v_d,
                ts2: v_xd,
                ts3: None,
                td: v_xd_sq_ns,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let xd_sq = Sra::sequence_output(&xd_sq_ns, &vec![SF_LOG as u64; num_outputs]);
        virtual_trace.extend(Sra::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sra,
                    ts1: v_xd_sq_ns,
                    ts2: v_sf_log,
                    ts3: None,
                    td: v_xd_sq,
                    imm: None,
                    virtual_sequence_remaining: Some(counter.get()),
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_ns))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq))),
                },
                advice_value: None,
            },
            virtual_slot,
        ));
        counter.subtract(Sra::SEQUENCE_LENGTH);

        // xd^2-1
        let xd_sq_minus1 = Sub::sequence_output(&xd_sq, &one);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Sub,
                ts1: v_xd_sq,
                ts2: v_one,
                ts3: None,
                td: v_xd_sq_minus1,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let gt_0: Vec<u64> = Gte::sequence_output(&xd_sq_minus1, &vec![0; num_outputs]);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Gte,
                ts1: v_xd_sq_minus1,
                ts2: None,
                ts3: None,
                td: v_gt0,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
                    if gt0 != 0 {
                        a_gt0
                    } else {
                        a_lt0
                    }
                },
            )
            .collect();
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Select,
                ts1: v_gt0,
                ts2: v_a_gt_0,
                ts3: v_a_lt_0,
                td: v_a,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let xd_cub_minusd_ns = Mul::sequence_output(&xd_sq_minus1, &d);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Mul,
                ts1: v_xd_sq_minus1,
                ts2: v_d,
                ts3: None,
                td: v_xd_cub_minusd_ns,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let xd_cub_minusd =
            Sra::sequence_output(&xd_cub_minusd_ns, &vec![SF_LOG as u64; num_outputs]);
        virtual_trace.extend(Sra::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sra,
                    ts1: v_xd_cub_minusd_ns,
                    ts2: v_sf_log,
                    ts3: None,
                    td: v_xd_cub_minusd,
                    imm: None,
                    virtual_sequence_remaining: Some(counter.get()),
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd_ns))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minusd))),
                },
                advice_value: None,
            },
            virtual_slot,
        ));
        counter.subtract(Sra::SEQUENCE_LENGTH);

        // ad(xd^2-1)
        let axd_cub_minusd_ns = Mul::sequence_output(&xd_cub_minusd, &a);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Mul,
                ts1: v_xd_cub_minusd,
                ts2: v_a,
                ts3: None,
                td: v_axd_cub_minusd_ns,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        let axd_cub_minusd =
            Sra::sequence_output(&axd_cub_minusd_ns, &vec![SF_LOG as u64; num_outputs]);
        virtual_trace.extend(Sra::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sra,
                    ts1: v_axd_cub_minusd_ns,
                    ts2: v_sf_log,
                    ts3: None,
                    td: v_axd_cub_minusd,
                    imm: None,
                    virtual_sequence_remaining: Some(counter.get()),
                    output_dims: cycle.instr.output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd_ns))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&sf_log))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minusd))),
                },
                advice_value: None,
            },
            virtual_slot,
        ));
        counter.subtract(Sra::SEQUENCE_LENGTH);

        // d + ad(xd^2-1)
        let approx: Vec<u64> = Add::sequence_output(&d, &axd_cub_minusd);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Add,
                ts1: v_d,
                ts2: v_axd_cub_minusd,
                ts3: None,
                td: v_approx,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualMove,
                ts1: v_approx,
                ts2: None,
                ts3: None,
                td,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
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

    fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}

fn rescale_down(q: i32) -> i32 {
    q >> SF_LOG
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;
    use num::pow::Pow;
    use rand::prelude::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Rsqrt>(ONNXOpcode::Rsqrt, 1);
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
            let result = Rsqrt::sequence_output(&[x], &[])[0];
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
