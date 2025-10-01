use crate::jolt::{
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    executor::instructions::{
        InstructionLookup, add::AddInstruction, mul::MulInstruction, relu::ReluInstruction,
        sub::SubInstruction, virtual_const::ConstInstruction,
    },
    lookup_table::LookupTables,
};
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    ProgramIO,
    graph::model::Model,
    tensor::Tensor,
    trace_types::{ONNXCycle, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

pub const WORD_SIZE: usize = 32;

pub fn trace<ModelFunc>(
    model: ModelFunc,
    input: &Tensor<i32>,
    preprocessing: &BytecodePreprocessing,
) -> (Vec<JoltONNXCycle>, ProgramIO)
where
    ModelFunc: Fn() -> Model,
{
    let (raw_trace, program_io) = onnx_tracer::execution_trace(model(), input);
    let trace = inline_tensor_trace(raw_trace, preprocessing);
    (trace, program_io)
}

/// # Note bytecode preprocessing specifies the bytecode trace since we don't prove sub-graphs
pub fn inline_tensor_trace(
    raw_trace: Vec<ONNXCycle>,
    preprocessing: &BytecodePreprocessing,
) -> Vec<JoltONNXCycle> {
    let mut trace = vec![JoltONNXCycle::no_op()];
    let mut current_pc = 1;

    for raw_cycle in raw_trace.iter() {
        let current_active_output_els = raw_cycle.instr.active_output_elements;
        let sequence = inline_tensor_cycle(
            raw_cycle,
            &preprocessing.bytecode[current_pc..current_pc + current_active_output_els],
        );
        trace.extend(sequence);
        current_pc += current_active_output_els;
    }
    println!("trace, {trace:#?}");
    trace.resize(preprocessing.code_size, JoltONNXCycle::no_op());
    trace
}

pub fn inline_tensor_cycle(
    raw_cycle: &ONNXCycle,
    instrs: &[JoltONNXBytecode],
) -> Vec<JoltONNXCycle> {
    let active_output_elements = raw_cycle.instr.active_output_elements;
    let tensor_values = TensorValues::from_cycle(raw_cycle, active_output_elements);

    (0..active_output_elements)
        .map(|i| {
            let memory_ops = MemoryOps {
                ts1_val: tensor_values.ts1_vals[i],
                ts2_val: tensor_values.ts2_vals[i],
                td_pre_val: tensor_values.td_pre_vals[i],
                td_post_val: tensor_values.td_post_vals[i],
            };
            let mut cycle = JoltONNXCycle {
                lookup: None,
                memory_ops,
            };
            cycle.construct_lookup_query(&instrs[i]);
            cycle
        })
        .collect()
}

struct TensorValues {
    ts1_vals: Vec<u64>,
    ts2_vals: Vec<u64>,
    td_pre_vals: Vec<u64>,
    td_post_vals: Vec<u64>,
}

impl TensorValues {
    fn from_cycle(raw_cycle: &ONNXCycle, size: usize) -> Self {
        let mut ts1_vals = raw_cycle.ts1_vals().unwrap_or_else(|| vec![0; size]);
        let mut ts2_vals = raw_cycle.ts2_vals().unwrap_or_else(|| vec![0; size]);

        // We need this because MatMults MCC do not follow the same pattern as other ops (it is handled separately) and we can safely store ts1 and ts2 as zero registers
        // TODO: Extend this match statement to all non-elementwise ops
        if raw_cycle.instr.opcode == ONNXOpcode::MatMult {
            ts1_vals = vec![0; size];
            ts2_vals = vec![0; size];
        }

        Self {
            ts1_vals,
            ts2_vals,
            td_pre_vals: raw_cycle.td_pre_vals().unwrap_or_else(|| vec![0; size]),
            td_post_vals: raw_cycle.td_post_vals().unwrap_or_else(|| vec![0; size]),
        }
    }
}

/// JoltONNXCycle's are paired with the preprocessed bytecode trace cycles
#[derive(Debug, Clone)]
pub struct JoltONNXCycle {
    /// Each instruction specifies an operation code that uniquely identifying its lookup function
    pub lookup: Option<LookupFunction>,
    /// register state
    pub memory_ops: MemoryOps,
}

impl JoltONNXCycle {
    pub fn new(lookup: Option<LookupFunction>, memory_ops: MemoryOps) -> Self {
        Self { lookup, memory_ops }
    }

    pub fn no_op() -> Self {
        Self {
            lookup: None,
            memory_ops: MemoryOps::default(),
        }
    }

    fn construct_lookup_query(&mut self, instr: &JoltONNXBytecode) {
        self.lookup = match instr.opcode {
            ONNXOpcode::Add => Some(LookupFunction::Add(AddInstruction::<WORD_SIZE>(
                self.memory_ops.ts1_val,
                self.memory_ops.ts2_val,
            ))),
            ONNXOpcode::Sub => Some(LookupFunction::Sub(SubInstruction::<WORD_SIZE>(
                self.memory_ops.ts1_val,
                self.memory_ops.ts2_val,
            ))),
            ONNXOpcode::Mul => Some(LookupFunction::Mul(MulInstruction::<WORD_SIZE>(
                self.memory_ops.ts1_val,
                self.memory_ops.ts2_val,
            ))),
            ONNXOpcode::Constant => Some(LookupFunction::Const(ConstInstruction::<WORD_SIZE>(
                instr.imm,
            ))),
            ONNXOpcode::Relu => Some(LookupFunction::Relu(ReluInstruction::<WORD_SIZE>(
                self.memory_ops.ts1_val,
            ))),
            _ => None,
        }
    }

    pub fn random(opcode: ONNXOpcode, rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        let memory_ops = MemoryOps {
            ts1_val: rng.next_u64(),
            ts2_val: rng.next_u64(),
            td_pre_val: rng.next_u64(),
            td_post_val: rng.next_u64(),
        };

        let jolt_onnx_bytecode = JoltONNXBytecode {
            opcode,
            imm: rng.next_u64(),
            ..JoltONNXBytecode::no_op()
        };
        let mut cycle = JoltONNXCycle::new(None, memory_ops);
        cycle.construct_lookup_query(&jolt_onnx_bytecode);
        cycle
    }

    pub fn ts1_read(&self) -> u64 {
        self.memory_ops.ts1_val
    }

    pub fn ts2_read(&self) -> u64 {
        self.memory_ops.ts2_val
    }

    /// # Returns
    /// (pre_val, post_val)
    pub fn td_write(&self) -> (u64, u64) {
        (self.memory_ops.td_pre_val, self.memory_ops.td_post_val)
    }
}

impl LookupQuery<WORD_SIZE> for JoltONNXCycle {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        self.lookup.as_ref().map_or((0, 0), |lookup_function| {
            lookup_function.to_instruction_inputs()
        })
    }

    fn to_lookup_index(&self) -> u64 {
        self.lookup
            .as_ref()
            .map_or(0, |lookup_function| lookup_function.to_lookup_index())
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        self.lookup.as_ref().map_or((0, 0), |lookup_function| {
            lookup_function.to_lookup_operands()
        })
    }

    fn to_lookup_output(&self) -> u64 {
        self.lookup
            .as_ref()
            .map_or(0, |lookup_function| lookup_function.to_lookup_output())
    }
}

// TODO: it seems we can refactor to get rid of this impl since JoltONNXBytecode implements this.
impl InstructionLookup<WORD_SIZE> for JoltONNXCycle {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        self.lookup
            .as_ref()
            .and_then(|lookup_function| lookup_function.lookup_table())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct MemoryOps {
    ts1_val: u64,
    ts2_val: u64,
    td_pre_val: u64,
    td_post_val: u64,
}

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl LookupQuery<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }

        impl InstructionLookup<$word_size> for $enum_name {
            fn lookup_table(&self) -> Option<LookupTables<$word_size>> {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.lookup_table(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum LookupFunction,
    const WORD_SIZE,
    Add: AddInstruction<WORD_SIZE>,
    Sub: SubInstruction<WORD_SIZE>,
    Mul: MulInstruction<WORD_SIZE>,
    Const: ConstInstruction<WORD_SIZE>,
    Relu: ReluInstruction<WORD_SIZE>,
);

/// This is only needed because runtime sometimes converts operands to
/// floating point for intermediate calculations, which can cause mismatches between
/// expected outputs and actual trace values.
#[cfg(test)]
pub fn sanity_check_mcc(
    bytecode: &[JoltONNXBytecode],
    execution_trace: &[JoltONNXCycle],
    K: usize,
) {
    let mut memory = vec![0u64; K];
    for (i, (cycle, instr)) in execution_trace.iter().zip(bytecode.iter()).enumerate() {
        // check reads
        let (ts1_addr, ts2_addr, td_addr) =
            (instr.ts1 as usize, instr.ts2 as usize, instr.td as usize);
        assert_eq!(
            memory[ts1_addr],
            cycle.ts1_read(),
            "TS1 READ error at cycle_{i}: {instr:#?} {cycle:#?}; Expected: {}, got: {} at address {ts1_addr} ",
            memory[ts1_addr] as u32 as i32,
            cycle.ts1_read() as u32 as i32
        );
        assert_eq!(
            memory[ts2_addr],
            cycle.ts2_read(),
            "TS2 READ error at cycle_{i}:{instr:#?} {cycle:#?}; Expected: {}, got: {} at address {ts2_addr} ",
            memory[ts2_addr] as u32 as i32,
            cycle.ts2_read() as u32 as i32
        );
        assert_eq!(
            memory[td_addr],
            cycle.td_write().0,
            "TD WRITE pre-state error at cycle_{i}: {instr:#?}{cycle:#?}; Expected: {}, got: {} at address {td_addr} ",
            memory[td_addr] as u32 as i32,
            cycle.td_write().0 as u32 as i32
        );
        // update memory
        memory[td_addr] = cycle.td_write().1;
    }
}
