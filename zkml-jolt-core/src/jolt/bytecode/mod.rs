use crate::jolt::{
    bytecode::{
        booleanity::BooleanitySumcheck, hamming_weight::HammingWeightSumcheck,
        read_raf_checking::ReadRafSumcheck,
    },
    dag::{stage::SumcheckStages, state_manager::StateManager},
    executor::instructions::{InstructionLookup, VirtualInstructionSequence, div::DivInstruction},
    lookup_table::{LookupTables, RangeCheckTable, ReLUTable},
    pcs::SumcheckId,
    sumcheck::SumcheckInstance,
    trace::{JoltONNXCycle, WORD_SIZE},
    witness::VirtualPolynomial,
};
use jolt_core::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial},
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        lookup_table::{
            equal::EqualTable, valid_div0::ValidDiv0Table,
            valid_signed_remainder::ValidSignedRemainderTable,
        },
        witness::{DTH_ROOT_OF_K, compute_d_parameter},
    },
};
use onnx_tracer::{
    graph::model::Model,
    tensor::Tensor,
    trace_types::{ONNXInstr, ONNXOpcode},
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    ops::{Index, IndexMut},
};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

pub const ZERO_ADDR_PREPEND: usize = 1; // TODO(Forpee): reserve output

pub mod booleanity;
pub mod hamming_weight;
pub mod read_raf_checking;

#[derive(Default)]
pub struct BytecodeDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for BytecodeDag
{
    fn stage4_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let (preprocessing, trace, _) = sm.get_prover_data();
        let bytecode_preprocessing = &preprocessing.shared.bytecode;

        let r_cycle: Vec<F> = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let E_1: Vec<F> = EqPolynomial::evals(&r_cycle);

        let F_1 = compute_ra_evals(bytecode_preprocessing, trace, &E_1);

        let read_raf = ReadRafSumcheck::new_prover(sm);
        let booleanity = BooleanitySumcheck::new_prover(sm, E_1, F_1.clone());
        let hamming_weight = HammingWeightSumcheck::new_prover(sm, F_1);

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_checking = ReadRafSumcheck::new_verifier(sm);
        let booleanity = BooleanitySumcheck::new_verifier(sm);
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

        vec![
            Box::new(read_checking),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }
}

#[inline(always)]
#[tracing::instrument(skip_all, name = "Bytecode::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    preprocessing: &BytecodePreprocessing,
    trace: &[JoltONNXCycle],
    eq_r_cycle: &[F],
) -> Vec<Vec<F>> {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);
    let log_K = preprocessing.code_size.log_2();
    let d = preprocessing.d;
    let log_K_chunk = log_K.div_ceil(d);
    let K_chunk = log_K_chunk.pow2();

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut j = chunk_index * chunk_size;
            for _ in trace_chunk {
                let mut pc = preprocessing.get_pc(j);
                for i in (0..d).rev() {
                    let k = pc % K_chunk;
                    result[i][k] += eq_r_cycle[j];
                    pc >>= log_K_chunk;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..d)
                    .map(|_| unsafe_allocate_zero_vec(K_chunk))
                    .collect::<Vec<_>>()
            },
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}

/// # Note: For our models (non-subgraph ones) bytecode trace is known up-front so we can preprocess it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<JoltONNXBytecode>,
    pub d: usize,
    pub memory_K: usize,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (opcode address, tensor sequence index, virtual sequence index or 0)
    pub tensor_virtual_pc_map: BTreeMap<(usize, usize, usize), usize>,
    /// Virtual tensor address map
    /// Maps (raw tensor address, tensor sequence index or 0) to virtual memory address
    pub vt_address_map: BTreeMap<(usize, usize), usize>,
    /// Used to expand the virtual trace
    pub max_td: usize,
    /// Get info of the bytecode from its td address
    pub td_lookup: HashMap<usize, ONNXInstr>,
    /// raw bytecode (used in precompiles)
    pub raw_bytecode: Vec<ONNXInstr>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// (Jolt-Optimized Unitary Logic Execution) bytecode line
pub struct JoltONNXBytecode {
    /// The unexpanded program counter (PC) address of this instruction in the ONNX binary bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// Index of the destination register for this instruction (0 if register is unused).
    pub td: u64,
    /// Index of the first source register for this instruction (0 if register is unused).
    pub ts1: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    pub ts2: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    pub ts3: u64,
    /// "Immediate" value for this instruction (0 if unused).
    pub imm: u64,
    /// Element-wise operations are decomposed into sequences of scalar instructions during preprocessing.
    /// This field tracks the remaining operations in such sequences for proper execution order.
    pub tensor_sequence_remaining: Option<usize>,
    pub virtual_sequence_remaining: Option<usize>,
}

impl JoltONNXBytecode {
    /// Used for padding
    pub fn no_op() -> Self {
        Self {
            address: 0,
            opcode: ONNXOpcode::Noop,
            td: 0,
            ts1: 0,
            ts2: 0,
            ts3: 0,
            imm: 0,
            tensor_sequence_remaining: None,
            virtual_sequence_remaining: None,
        }
    }

    /// Effectively a no-op but is not placed at address 0
    pub fn addressed_no_op(address: usize) -> Self {
        Self {
            address,
            opcode: ONNXOpcode::AddressedNoop,
            td: 0,
            ts1: 0,
            ts2: 0,
            ts3: 0,
            imm: 0,
            tensor_sequence_remaining: None,
            virtual_sequence_remaining: None,
        }
    }

    #[rustfmt::skip]
    pub fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsTs1Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualMove
            | ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
            | ONNXOpcode::Gte
            | ONNXOpcode::Relu
            | ONNXOpcode::Output
            | ONNXOpcode::Broadcast
        );

        flags[CircuitFlags::RightOperandIsTs2Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
            | ONNXOpcode::Gte
        );

        flags[CircuitFlags::RightOperandIsImm as usize] = matches!(
            self.opcode,
            | ONNXOpcode::VirtualMove
        );

        flags[CircuitFlags::AddOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::VirtualMove
            | ONNXOpcode::Relu
            | ONNXOpcode::Output
            | ONNXOpcode::Broadcast
        );

        flags[CircuitFlags::SubtractOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Sub,
        );

        flags[CircuitFlags::MultiplyOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Mul,
        );

        flags[CircuitFlags::WriteLookupOutputToTD as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualAdvice
            | ONNXOpcode::VirtualMove
            | ONNXOpcode::VirtualConst
            | ONNXOpcode::Gte
            | ONNXOpcode::Relu
            | ONNXOpcode::Output
            | ONNXOpcode::Broadcast
        );

        flags[CircuitFlags::Advice as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAdvice
        );

        flags[CircuitFlags::Const as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualConst
            | ONNXOpcode::Constant
        );

        flags[CircuitFlags::Assert as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
        );

        flags[CircuitFlags::Assert as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
        );

        flags[CircuitFlags::IsNoop as usize] = matches!(
            self.opcode,
            ONNXOpcode::Noop
        );

        flags[CircuitFlags::Select as usize] = matches!(
            self.opcode,
            ONNXOpcode::Select
        );

        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            !(self.tensor_sequence_remaining.unwrap_or(0) == 0 && self.virtual_sequence_remaining.unwrap_or(0) == 0);

        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.tensor_sequence_remaining.is_some() || self.virtual_sequence_remaining.is_some();

        flags
    }
}

impl InstructionLookup<WORD_SIZE> for JoltONNXBytecode {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self.opcode {
            ONNXOpcode::Add => Some(RangeCheckTable.into()),
            ONNXOpcode::Sub => Some(RangeCheckTable.into()),
            ONNXOpcode::Mul => Some(RangeCheckTable.into()),
            ONNXOpcode::Constant => Some(RangeCheckTable.into()),
            ONNXOpcode::Relu => Some(ReLUTable.into()),
            ONNXOpcode::VirtualConst => Some(RangeCheckTable.into()),
            ONNXOpcode::VirtualAdvice => Some(RangeCheckTable.into()),
            ONNXOpcode::VirtualMove => Some(RangeCheckTable.into()),
            ONNXOpcode::VirtualAssertValidSignedRemainder => Some(ValidSignedRemainderTable.into()),
            ONNXOpcode::VirtualAssertValidDiv0 => Some(ValidDiv0Table.into()),
            ONNXOpcode::VirtualAssertEq => Some(EqualTable.into()),
            ONNXOpcode::Broadcast => Some(RangeCheckTable.into()),
            _ => None,
        }
    }
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess<ModelFunc>(model: ModelFunc) -> Self
    where
        ModelFunc: Fn() -> Model,
    {
        let (mut bytecode, memory_K, vt_address_map, max_td, td_lookup, raw_bytecode) =
            Self::inline_tensor_instrs(model);
        // Append an addressed no-op instruction at the end to simplify the PC logic in the VM
        bytecode.push(JoltONNXBytecode::addressed_no_op(
            bytecode.last().map_or(0, |cycle| cycle.address) + 1,
        ));
        let mut tensor_virtual_pc_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            assert_eq!(
                tensor_virtual_pc_map.insert(
                    (
                        instruction.address,
                        instruction.tensor_sequence_remaining.unwrap_or(0),
                        instruction.virtual_sequence_remaining.unwrap_or(0),
                    ),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, JoltONNXBytecode::no_op());
        assert_eq!(tensor_virtual_pc_map.insert((0, 0, 0), 0), None);
        let d = compute_d_parameter(bytecode.len().next_power_of_two());
        // Make log(code_size) a multiple of d
        let code_size = (bytecode.len().next_power_of_two().log_2().div_ceil(d) * d)
            .pow2()
            .max(DTH_ROOT_OF_K);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, JoltONNXBytecode::no_op());
        Self {
            code_size,
            bytecode,
            d,
            memory_K,
            tensor_virtual_pc_map,
            vt_address_map,
            max_td,
            td_lookup,
            raw_bytecode,
        }
    }

    /// Getter for td_lookup
    pub fn td_lookup(&self) -> &HashMap<usize, ONNXInstr> {
        &self.td_lookup
    }

    pub fn inline_tensor_instrs<ModelFunc>(model: ModelFunc) -> RawToJoltResult
    where
        ModelFunc: Fn() -> Model,
    {
        let raw_bytecode = onnx_tracer::decode_model(model());
        // Build a lookup map for O(1) instruction lookups by td value
        let td_lookup: HashMap<usize, ONNXInstr> = raw_bytecode
            .iter()
            .filter_map(|instr| instr.td.map(|td| (td, instr.clone())))
            .collect();
        // Get largest td value
        let max_td = raw_bytecode
            .iter()
            .filter_map(|instr| instr.td)
            .max()
            .unwrap_or(0);
        let raw_bytecode = Self::expand_raw_bytecode(raw_bytecode, max_td);

        let mut preprocessed_bytecode: Vec<JoltONNXBytecode> = Vec::new();

        // Memory management and instruction preprocessing:
        // 1. Allocate virtual memory addresses for tensor operands and results
        // 2. Decompose tensor operations into scalar instructions with individual immediate values
        // 3. Map ONNX instruction operands to virtual register addresses for the Jolt VM

        // virtual tensor address
        let mut vt_address: usize = 0;
        // virtual tensor address map
        let mut vt_address_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let max_active_output_elements = max_output_elements(&raw_bytecode);

        // reserve address space for zero registers
        for i in 0..max_active_output_elements {
            vt_address_map.insert((0, max_active_output_elements - i - 1), vt_address);
            vt_address += 1;
        }

        // convert each ONNX instruction to one or more jolt bytecode lines
        // and allocate virtual memory addresses for their operands and results
        // also update the virtual tensor address map
        // to map ONNX tensor addresses to Jolt VM virtual addresses
        // e.g., ONNX tensor address 0 (zero register) maps to virtual addresses 0..max_active_output_elements
        // e.g., ONNX tensor address 1 maps to virtual addresses max_active_output_elements..(max_active_output_elements + its size)
        // etc.
        for instruction in raw_bytecode.iter() {
            let jolt_instructions = raw_to_jolt_bytecode(
                instruction,
                &mut vt_address,
                &mut vt_address_map,
                &td_lookup,
            );
            preprocessed_bytecode.extend(jolt_instructions);
        }
        (
            preprocessed_bytecode,
            vt_address.next_power_of_two(),
            vt_address_map,
            max_td,
            td_lookup,
            raw_bytecode,
        )
    }

    pub fn get_pc(&self, i: usize) -> usize {
        *self
            .tensor_virtual_pc_map
            .get(&(
                self.bytecode[i].address,
                self.bytecode[i].tensor_sequence_remaining.unwrap_or(0),
                self.bytecode[i].virtual_sequence_remaining.unwrap_or(0),
            ))
            .unwrap()
    }

    /// Expand the virtual instructions of the raw ONNX bytecode.
    ///
    /// # Parameters
    ///
    /// * `raw_bytecode` - The raw ONNX bytecode to be expanded
    /// * `max_td` - used to calculate a unique register address for virtual registers used in virtual instructions
    fn expand_raw_bytecode(raw_bytecode: Vec<ONNXInstr>, max_td: usize) -> Vec<ONNXInstr> {
        raw_bytecode
            .into_iter()
            .flat_map(|instr| match instr.opcode {
                ONNXOpcode::Div => DivInstruction::<32>::virtual_sequence(instr, max_td),
                _ => vec![instr],
            })
            .collect()
    }

    /// Getter for raw bytecode
    pub fn raw_bytecode(&self) -> &[ONNXInstr] {
        &self.raw_bytecode
    }

    /// Collects memory addresses for tensor elements based on an instruction.
    ///
    /// This helper method extracts the memory addresses for all active output elements
    /// of a given instruction by querying the bytecode preprocessing map.
    ///
    /// # Parameters
    ///
    /// * `instr` - The ONNX instruction containing tensor information
    /// * `bytecode_preprocessing` - Contains the mapping from virtual addresses to physical addresses
    ///
    /// # Returns
    ///
    /// A vector of memory addresses for the instruction's active output elements
    pub fn collect_addresses(&self, instr: &ONNXInstr) -> Vec<usize> {
        (0..instr.active_output_elements)
            .map(|i| {
                self.vt_address_map[&(
                    zkvm_address(instr.td),
                    tensor_sequence_remaining(instr.active_output_elements, i),
                )]
            })
            .collect()
    }
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, PartialEq, EnumCountMacro, EnumIter, Eq, Hash, PartialOrd, Ord)]
pub enum CircuitFlags {
    /// 1 if the first instruction operand is TS1 value; 0 otherwise.
    LeftOperandIsTs1Value,
    /// 1 if the first instruction operand is TS2 value; 0 otherwise.
    RightOperandIsTs2Value,
    /// 1 if the second instruction operand is `imm`; 0 otherwise.
    RightOperandIsImm,
    /// 1 if the first lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// 1 if the first lookup operand is the difference between the two instruction operands.
    SubtractOperands,
    /// 1 if the first lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    /// 1 if the lookup output is to be stored in `td` at the end of the step.
    WriteLookupOutputToTD,
    /// 1 if the instruction is "inline", as defined in Section 6.1 of the Jolt paper.
    InlineSequenceInstruction,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdateUnexpandedPC,
    /// Is (virtual) advice instruction
    Advice,
    /// 1 if this is constant instruction; 0 otherwise.
    Const,
    /// Is noop instruction
    IsNoop,
    /// 1 if this is the select operator
    Select,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for [bool; NUM_CIRCUIT_FLAGS] {
    fn is_interleaved_operands(&self) -> bool {
        !self[CircuitFlags::AddOperands]
            && !self[CircuitFlags::SubtractOperands]
            && !self[CircuitFlags::MultiplyOperands]
            && !self[CircuitFlags::Advice]
            && !self[CircuitFlags::Const]
    }
}

impl Index<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    type Output = bool;
    fn index(&self, index: CircuitFlags) -> &bool {
        &self[index as usize]
    }
}

impl IndexMut<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    fn index_mut(&mut self, index: CircuitFlags) -> &mut bool {
        &mut self[index as usize]
    }
}

pub type RawToJoltResult = (
    Vec<JoltONNXBytecode>,
    usize,
    BTreeMap<(usize, usize), usize>,
    usize,
    HashMap<usize, ONNXInstr>,
    Vec<ONNXInstr>,
);

/// Convert a raw ONNX instruction to one or more Jolt bytecode instructions.
/// This is done by decomposing tensor operations into scalar instructions
/// and mapping tensor addresses to virtual memory addresses.
pub fn raw_to_jolt_bytecode(
    raw: &ONNXInstr,
    vt_address: &mut usize,
    vt_address_map: &mut BTreeMap<(usize, usize), usize>,
    td_lookup: &HashMap<usize, ONNXInstr>,
) -> Vec<JoltONNXBytecode> {
    let mut jolt_instructions: Vec<JoltONNXBytecode> = vec![];
    // Address allocation strategy:
    // 1. Reserve memory slots for zero registers (max active output elements across all instructions)
    // 2. For each scalar operation derived from tensor decomposition:
    //    - ts1/ts2: Read from previously allocated addresses (from prior instruction outputs)
    //    - td: Allocate new virtual memory address for this instruction's result
    // 3. Map ONNX tensor addresses to virtual memory addresses for Jolt VM execution

    let active_output_elements = raw.active_output_elements;

    // get ts1 and ts2 addresses
    let (vts1, vts2, vts3) = match raw.opcode {
        ONNXOpcode::MatMult | ONNXOpcode::Sum => {
            // We need this because MatMults MCC do not follow the same pattern as other ops (it is handled separately) and we can safely store ts1 and ts2 as zero registers
            (
                vec![0; active_output_elements],
                vec![0; active_output_elements],
                vec![0; active_output_elements],
            )
        }
        ONNXOpcode::Broadcast => {
            // Get the operand instruction
            let operand_instr = td_lookup
                .get(&raw.ts1.unwrap())
                .unwrap_or_else(|| panic!("Missing instruction for td {}", raw.ts1.unwrap()));

            // map the ts1 addresses to the broadcasted addresses
            let vts1 = (0..operand_instr.active_output_elements)
                .map(|i| {
                    vt_address_map[&(
                        zkvm_address(operand_instr.td),
                        tensor_sequence_remaining(operand_instr.active_output_elements, i),
                    )]
                })
                .collect::<Vec<usize>>();
            // convert it to a tensor
            let vts1_tensor = Tensor::new(
                Some(&vts1.iter().map(|&x| x as i32).collect::<Vec<i32>>()),
                &operand_instr.output_dims,
            )
            .unwrap();
            // broadcast it to the new shape
            let broadcasted_tensor = vts1_tensor.expand(&raw.output_dims).unwrap();
            // flatten it back to a vector
            let vts1 = broadcasted_tensor
                .data()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>();
            (
                vts1,
                vec![0; active_output_elements], // ts2 is unused in broadcast
                vec![0; active_output_elements], // ts3 is unused in broadcast
            )
        }
        _ => {
            // Helper function to map tensor sequence addresses
            let map_tensor_addresses = |ts_opt: Option<usize>| -> Vec<usize> {
                (0..active_output_elements)
                    .map(|i| {
                        vt_address_map[&(
                            zkvm_address(ts_opt),
                            tensor_sequence_remaining(active_output_elements, i),
                        )]
                    })
                    .collect()
            };
            let vts1 = map_tensor_addresses(raw.ts1);
            let vts2 = map_tensor_addresses(raw.ts2);
            let vts3 = map_tensor_addresses(raw.ts3);
            (vts1, vts2, vts3)
        }
    };

    // calculate td address
    let vtd = (0..active_output_elements)
        .map(|i| {
            if raw.td.is_some() {
                let addr = *vt_address;
                assert_eq!(
                    vt_address_map.insert(
                        (
                            zkvm_address(raw.td),
                            tensor_sequence_remaining(active_output_elements, i)
                        ),
                        addr
                    ),
                    None,
                );
                *vt_address += 1;
                addr
            } else {
                i
            }
        })
        .collect::<Vec<usize>>();

    let imm = raw.imm().unwrap_or(vec![0; active_output_elements]);
    for i in 0..active_output_elements {
        jolt_instructions.push(JoltONNXBytecode {
            address: raw.address,
            opcode: raw.opcode.clone(),
            td: vtd[i] as u64,
            ts1: vts1[i] as u64,
            ts2: vts2[i] as u64,
            ts3: vts3[i] as u64,
            imm: imm[i],
            tensor_sequence_remaining: Some(tensor_sequence_remaining(active_output_elements, i)),
            virtual_sequence_remaining: raw.virtual_sequence_remaining,
        });
    }
    jolt_instructions
}

/// Returns the maximum number of active output elements across all instructions in the bytecode.
/// This is used to determine the size of the zero register space that needs to be reserved in memory.
pub fn max_output_elements(bytecode: &[ONNXInstr]) -> usize {
    bytecode
        .iter()
        .map(|instr| instr.active_output_elements)
        .max()
        .unwrap_or(1)
}

/// Convert the raw pc to the zkvm address by prepending space for the zero register
#[inline]
pub fn zkvm_address(t: Option<usize>) -> usize {
    t.map_or(0, |t| t + ZERO_ADDR_PREPEND)
}

/// Given the number of active output elements and the current index,
/// returns the remaining number of elements in the tensor sequence.
/// This is used to track the progress of tensor operations that have been
/// decomposed into scalar instructions.
///
/// # Parameters
/// - `active_output_elements`: Total number of active output elements in the tensor.
/// - `current_index`: The current index in the tensor sequence (0-based).
///
/// # Returns
/// The number of remaining elements in the tensor sequence after the current index.
#[inline]
pub fn tensor_sequence_remaining(active_output_elements: usize, current_index: usize) -> usize {
    active_output_elements - current_index - 1
}
