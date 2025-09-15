use crate::jolt::{
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    executor::instructions::{
        add::AddInstruction, mul::MulInstruction, sub::SubInstruction,
        virtual_const::ConstInstruction,
    },
};
use jolt_core::zkvm::{
    instruction::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};
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
    let current_active_output_els = raw_trace[0].instr.active_output_elements;
    for raw_cycle in raw_trace.iter() {
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
    let ts1_vals = raw_cycle
        .ts1_vals()
        .unwrap_or(vec![0; active_output_elements]);
    let ts2_vals = raw_cycle
        .ts2_vals()
        .unwrap_or(vec![0; active_output_elements]);
    let td_pre_vals = raw_cycle
        .td_pre_vals()
        .unwrap_or(vec![0; active_output_elements]);
    let td_post_vals = raw_cycle
        .td_post_vals()
        .unwrap_or(vec![0; active_output_elements]);
    (0..active_output_elements)
        .map(|i| {
            let mut cycle = JoltONNXCycle {
                lookup: None,
                memory_ops: MemoryOps {
                    ts1_val: ts1_vals[i],
                    ts2_val: ts2_vals[i],
                    td_pre_val: td_pre_vals[i],
                    td_post_val: td_post_vals[i],
                },
            };
            cycle.construct_lookup_query(&instrs[i]);
            cycle
        })
        .collect()
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
            _ => None,
        }
    }

    pub fn ts1_read(&self) -> u64 {
        self.memory_ops.ts1_val
    }

    pub fn ts2_read(&self) -> u64 {
        self.memory_ops.ts2_val
    }

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
        trait $trait_name:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl $trait_name<$word_size> for $enum_name {
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
    trait LookupQuery,
    Add: AddInstruction<WORD_SIZE>,
    Sub: SubInstruction<WORD_SIZE>,
    Mul: MulInstruction<WORD_SIZE>,
    Const: ConstInstruction<WORD_SIZE>,
);
