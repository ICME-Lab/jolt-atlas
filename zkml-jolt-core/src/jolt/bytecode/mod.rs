use crate::jolt::{
    bytecode::read_raf_checking::ReadRafCheck,
    dag::{stage::SumcheckStages, state_manager::StateManager},
    executor::instructions::InstructionLookup,
    lookup_table::{LookupTables, RangeCheckTable, ReLUTable},
    sumcheck::SumcheckInstance,
    trace::WORD_SIZE,
};
use jolt_core::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
    zkvm::lookup_table::{
        equal::EqualTable, pow2::Pow2Table, shift_right_bitmask::ShiftRightBitmaskTable,
        signed_greater_than_equal::SignedGreaterThanEqualTable, valid_div0::ValidDiv0Table,
        valid_signed_remainder::ValidSignedRemainderTable, virtual_sra::VirtualSRATable,
    },
};
use onnx_tracer::{
    graph::model::Model,
    tensor::Tensor,
    trace_types::{AtlasInstr, AtlasOpcode, ONNXInstr},
    utils::VirtualSlotCounter,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    ops::{Index, IndexMut},
};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

pub const ZERO_ADDR_PREPEND: usize = 1; // TODO(AntoineF4C5): reserve output

pub mod read_raf_checking;

#[derive(Default)]
pub struct BytecodeDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for BytecodeDag
{
    fn stage6_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        ReadRafCheck::prove(sm);
        vec![]
    }

    fn stage6_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        ReadRafCheck::verify(sm);
        vec![]
    }
}

/// # Note: For our models (non-subgraph ones) bytecode trace is known up-front so we can preprocess it
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<JoltONNXBytecode>,
    pub memory_K: usize,
    /// Virtual tensor address map
    /// Maps `(zkvm tensor address, remaining tensor sequence elements)` to unique virtual
    /// memory locations, ensuring every scalar element produced during inlining has a stable
    /// address once tensors are decomposed into per-element instructions.
    pub vt_address_map: BTreeMap<(usize, usize), usize>,
    /// Used to expand the virtual trace
    pub max_td: usize,
    /// Get info of the bytecode from its td address
    pub td_lookup: HashMap<usize, AtlasInstr>,
    /// raw bytecode (used in precompiles)
    pub raw_bytecode: Vec<AtlasInstr>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess<ModelFunc>(model: ModelFunc) -> Self
    where
        ModelFunc: Fn() -> Model,
    {
        let (mut bytecode, memory_K, vt_address_map, max_td, td_lookup, raw_bytecode) =
            Self::inline_tensor_instrs(model);
        Self::finalize_bytecode(&mut bytecode);
        let code_size = Self::compute_code_size(bytecode.len());
        Self {
            code_size,
            bytecode,
            memory_K,
            vt_address_map,
            max_td,
            td_lookup,
            raw_bytecode,
        }
    }

    /// Finalizes bytecode by adding padding, no-ops, and setting the halt flag.
    fn finalize_bytecode(bytecode: &mut Vec<JoltONNXBytecode>) {
        Self::prepend_noop(bytecode);
        Self::pad_to_power_of_two(bytecode);
        Self::mark_halt(bytecode);
    }

    /// Adds no-op instructions at the beginning of bytecode.
    fn prepend_noop(bytecode: &mut Vec<JoltONNXBytecode>) {
        bytecode.insert(0, JoltONNXBytecode::no_op());
    }

    /// Pads bytecode to the next power of 2 with addressed no-ops.
    fn pad_to_power_of_two(bytecode: &mut Vec<JoltONNXBytecode>) {
        let target_size = Self::compute_code_size(bytecode.len());
        let current_len = bytecode.len();
        let mut pc = bytecode
            .last()
            .expect("Bytecode should not be empty after adding boundary no-ops")
            .address;
        bytecode.extend((current_len..target_size).map(|_| {
            pc += 1;
            JoltONNXBytecode::addressed_no_op(pc)
        }));
    }

    /// Marks the last instruction in bytecode as a halt instruction.
    fn mark_halt(bytecode: &mut [JoltONNXBytecode]) {
        bytecode
            .last_mut()
            .expect("Bytecode should not be empty after finalization")
            .halt = true;
    }

    /// Computes the final code size as a power of 2.
    ///
    /// # Parameters
    /// * `bytecode_len` - The current length of the bytecode
    ///
    /// # Returns
    /// The code size as a power of 2, with a minimum of 32 since twist at its current state requires trace length >= 32
    fn compute_code_size(bytecode_len: usize) -> usize {
        const MIN_CODE_SIZE: usize = 1 << 5;
        bytecode_len.next_power_of_two().max(MIN_CODE_SIZE)
    }

    /// Getter for td_lookup
    pub fn td_lookup(&self) -> &HashMap<usize, AtlasInstr> {
        &self.td_lookup
    }

    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::inline_tensor_instrs")]
    pub fn inline_tensor_instrs<ModelFunc>(model: ModelFunc) -> RawToJoltResult
    where
        ModelFunc: Fn() -> Model,
    {
        let (expanded_bytecode, max_td) = Self::decode_and_expand_model(model);
        let td_lookup = Self::build_td_lookup(&expanded_bytecode);

        // Memory management and instruction preprocessing:
        // 1. Allocate virtual memory addresses for tensor operands and results
        // 2. Decompose tensor operations into scalar instructions with individual immediate values
        // 3. Map ONNX instruction operands to virtual register addresses for the Jolt VM
        let max_output_elements = max_output_elements(&expanded_bytecode);
        let mut inliner = BytecodeInstructionInliner::new(max_output_elements, &td_lookup);

        // Inline every instruction while recording the virtual tensor map.
        let preprocessed_bytecode: Vec<JoltONNXBytecode> = expanded_bytecode
            .iter()
            .flat_map(|instruction| inliner.inline_instruction(instruction))
            .collect();

        let (vt_address_map, next_virtual_address) = inliner.finish();

        (
            preprocessed_bytecode,
            next_virtual_address.next_power_of_two(),
            vt_address_map,
            max_td,
            td_lookup,
            expanded_bytecode,
        )
    }

    /// Decodes the model into ONNX instructions, expands virtual instructions, and returns the
    /// expanded trace along with the maximum td encountered. The max_td value is computed on the
    /// unexpanded trace so virtual instructions can reserve unique register addresses deterministically.
    fn decode_and_expand_model<ModelFunc>(model: ModelFunc) -> (Vec<AtlasInstr>, usize)
    where
        ModelFunc: Fn() -> Model,
    {
        let decoded_bytecode = onnx_tracer::decode_model(model());
        let max_td = decoded_bytecode
            .iter()
            .filter_map(|instr| instr.td)
            .max()
            .unwrap_or(0);
        let expanded_bytecode = Self::expand_virtual_bytecode(decoded_bytecode, max_td);
        (expanded_bytecode, max_td)
    }

    /// Build a lookup map for O(1) instruction lookups by td value. Used in the precompiles to
    /// get the i/o addresses of the precompile operation.
    fn build_td_lookup(bytecode: &[AtlasInstr]) -> HashMap<usize, AtlasInstr> {
        bytecode
            .iter()
            .filter_map(|instr| instr.td.map(|td| (td, instr.clone())))
            .collect()
    }

    pub fn get_pc(&self, i: usize) -> usize {
        i
    }

    /// Expand the virtual instructions of the raw ONNX bytecode.
    ///
    /// # Parameters
    ///
    /// * `raw_bytecode` - The raw ONNX bytecode to be expanded
    /// * `max_td` - used to calculate a unique register address for virtual registers used in virtual instructions
    fn expand_virtual_bytecode(raw_bytecode: Vec<ONNXInstr>, max_td: usize) -> Vec<AtlasInstr> {
        let mut virtual_slot_counter = VirtualSlotCounter::new(max_td + 1);
        raw_bytecode
            .into_iter()
            .flat_map(|instr| instr.virtual_sequence(&mut virtual_slot_counter))
            .collect()
    }

    /// Getter for raw bytecode
    pub fn raw_bytecode(&self) -> &[AtlasInstr] {
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
    pub fn collect_addresses(&self, instr: &AtlasInstr) -> Vec<usize> {
        (0..instr.num_output_elements())
            .map(|i| {
                self.vt_address_map[&(
                    zkvm_address(instr.td),
                    tensor_sequence_remaining(instr.num_output_elements(), i),
                )]
            })
            .collect()
    }
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

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
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Is (virtual) advice instruction
    Advice,
    /// 1 if this is constant instruction; 0 otherwise.
    Const,
    /// 1 if this is the select operator
    Select,
    /// 1 if this is a halt instruction
    Halt,
}

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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
/// (Jolt-Optimized Unitary Logic Execution) bytecode line
pub struct JoltONNXBytecode {
    /// The unexpanded program counter (PC) address of this instruction in the ONNX binary bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: AtlasOpcode,
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
    /// Indicates whether this instruction is a halting instruction.
    pub halt: bool,
}

impl JoltONNXBytecode {
    /// Used for padding
    pub fn no_op() -> Self {
        Self {
            opcode: AtlasOpcode::Noop,
            ..Default::default()
        }
    }

    /// Effectively a no-op but is not placed at address 0
    pub fn addressed_no_op(address: usize) -> Self {
        Self {
            address,
            opcode: AtlasOpcode::AddressedNoop,
            ..Self::no_op()
        }
    }

    #[rustfmt::skip]
    pub fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsTs1Value as usize] = matches!(
            self.opcode,
            AtlasOpcode::Add
            | AtlasOpcode::Broadcast
            | AtlasOpcode::Eq
            | AtlasOpcode::Gte
            | AtlasOpcode::Mul
            // | AtlasOpcode::Output
            | AtlasOpcode::Relu
            | AtlasOpcode::Reshape
            | AtlasOpcode::Sub
            | AtlasOpcode::VirtualAssertEq
            | AtlasOpcode::VirtualAssertValidDiv0
            | AtlasOpcode::VirtualAssertValidSignedRemainder
            | AtlasOpcode::VirtualMove
            | AtlasOpcode::VirtualPow2
            | AtlasOpcode::VirtualShiftRightBitmask
            | AtlasOpcode::VirtualSra
        );

        flags[CircuitFlags::RightOperandIsTs2Value as usize] = matches!(
            self.opcode,
            AtlasOpcode::Add
            | AtlasOpcode::Eq
            | AtlasOpcode::Gte
            | AtlasOpcode::Mul
            | AtlasOpcode::Sub
            | AtlasOpcode::VirtualAssertEq
            | AtlasOpcode::VirtualAssertValidDiv0
            | AtlasOpcode::VirtualAssertValidSignedRemainder
            | AtlasOpcode::VirtualSra
        );

        flags[CircuitFlags::RightOperandIsImm as usize] = matches!(
            self.opcode,
            | AtlasOpcode::VirtualMove
        );

        flags[CircuitFlags::AddOperands as usize] = matches!(
            self.opcode,
            AtlasOpcode::Add
            | AtlasOpcode::Broadcast
            // | AtlasOpcode::Output
            | AtlasOpcode::Relu
            | AtlasOpcode::Reshape
            | AtlasOpcode::VirtualMove
            | AtlasOpcode::VirtualPow2
            | AtlasOpcode::VirtualShiftRightBitmask
        );

        flags[CircuitFlags::SubtractOperands as usize] = matches!(
            self.opcode,
            AtlasOpcode::Sub,
        );

        flags[CircuitFlags::MultiplyOperands as usize] = matches!(
            self.opcode,
            AtlasOpcode::Mul,
        );

        flags[CircuitFlags::WriteLookupOutputToTD as usize] = matches!(
            self.opcode,
            AtlasOpcode::Add
            | AtlasOpcode::Broadcast
            | AtlasOpcode::Eq
            | AtlasOpcode::Gte
            // | AtlasOpcode::Output
            | AtlasOpcode::Mul
            | AtlasOpcode::Relu
            | AtlasOpcode::Reshape
            | AtlasOpcode::Sub
            | AtlasOpcode::VirtualAdvice
            | AtlasOpcode::VirtualMove
            | AtlasOpcode::VirtualPow2
            | AtlasOpcode::VirtualShiftRightBitmask
            | AtlasOpcode::VirtualSra
        );

        flags[CircuitFlags::Advice as usize] = matches!(
            self.opcode,
            AtlasOpcode::VirtualAdvice
        );

        flags[CircuitFlags::Const as usize] = matches!(
            self.opcode,
            AtlasOpcode::Constant
        );

        flags[CircuitFlags::Assert as usize] = matches!(
            self.opcode,
            AtlasOpcode::VirtualAssertEq
            | AtlasOpcode::VirtualAssertValidDiv0
            | AtlasOpcode::VirtualAssertValidSignedRemainder
        );

        flags[CircuitFlags::Select as usize] = matches!(
            self.opcode,
            AtlasOpcode::Select
        );

        flags[CircuitFlags::Halt as usize] = self.halt;

        flags
    }
}

// TODO(AntoineF4C5): Get rid of this helper
impl InstructionLookup<WORD_SIZE> for JoltONNXBytecode {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self.opcode {
            AtlasOpcode::Add => Some(RangeCheckTable.into()),
            AtlasOpcode::Broadcast => Some(RangeCheckTable.into()),
            AtlasOpcode::Constant => Some(RangeCheckTable.into()),
            AtlasOpcode::Eq => Some(EqualTable.into()),
            AtlasOpcode::Gte => Some(SignedGreaterThanEqualTable.into()),
            AtlasOpcode::Mul => Some(RangeCheckTable.into()),
            AtlasOpcode::Relu => Some(ReLUTable.into()),
            AtlasOpcode::Reshape => Some(RangeCheckTable.into()),
            AtlasOpcode::Sub => Some(RangeCheckTable.into()),
            AtlasOpcode::VirtualAssertEq => Some(EqualTable.into()),
            AtlasOpcode::VirtualAssertValidDiv0 => Some(ValidDiv0Table.into()),
            AtlasOpcode::VirtualAssertValidSignedRemainder => {
                Some(ValidSignedRemainderTable.into())
            }
            AtlasOpcode::VirtualAdvice => Some(RangeCheckTable.into()),
            AtlasOpcode::VirtualMove => Some(RangeCheckTable.into()),
            AtlasOpcode::VirtualPow2 => Some(Pow2Table.into()),
            AtlasOpcode::VirtualSra => Some(VirtualSRATable.into()),
            AtlasOpcode::VirtualShiftRightBitmask => Some(ShiftRightBitmaskTable.into()),
            _ => None,
        }
    }
}

pub type RawToJoltResult = (
    Vec<JoltONNXBytecode>,
    usize,
    BTreeMap<(usize, usize), usize>,
    usize,
    HashMap<usize, AtlasInstr>,
    Vec<AtlasInstr>,
);

/// Helper responsible for converting raw ONNX instructions into
/// scalar Jolt bytecode while maintaining the virtual tensor address map.
struct BytecodeInstructionInliner<'a> {
    td_lookup: &'a HashMap<usize, AtlasInstr>,
    allocator: VirtualTensorAllocator,
}

impl<'a> BytecodeInstructionInliner<'a> {
    fn new(max_zero_register_span: usize, td_lookup: &'a HashMap<usize, AtlasInstr>) -> Self {
        Self {
            td_lookup,
            allocator: VirtualTensorAllocator::new(max_zero_register_span),
        }
    }

    fn inline_instruction(&mut self, raw: &AtlasInstr) -> Vec<JoltONNXBytecode> {
        let element_count = raw.num_output_elements();
        let (ts1, ts2, ts3) = self.resolve_sources(raw, element_count);
        let td = self.allocate_destinations(raw.td, element_count);
        let immediates = raw.imm().unwrap_or_else(|| vec![0; element_count]);

        (0..element_count)
            .map(|index| JoltONNXBytecode {
                address: raw.address,
                opcode: raw.opcode.clone(),
                td: td[index] as u64,
                ts1: ts1[index] as u64,
                ts2: ts2[index] as u64,
                ts3: ts3[index] as u64,
                imm: immediates[index],
                tensor_sequence_remaining: Some(tensor_sequence_remaining(element_count, index)),
                virtual_sequence_remaining: raw.virtual_sequence_remaining,
                halt: false,
            })
            .collect()
    }

    fn finish(self) -> (BTreeMap<(usize, usize), usize>, usize) {
        self.allocator.finalize()
    }

    // TODO(AntoineF4C5): Have behavior defined in each instruction definition
    fn resolve_sources(
        &self,
        raw: &AtlasInstr,
        element_count: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        match raw.opcode {
            AtlasOpcode::Einsum(_) | AtlasOpcode::Sum(_) => (
                vec![0; element_count],
                vec![0; element_count],
                vec![0; element_count],
            ),
            AtlasOpcode::Broadcast => (
                self.broadcast_addresses(raw, element_count),
                vec![0; element_count],
                vec![0; element_count],
            ),
            _ => (
                self.allocator.sequence_addresses(raw.ts1, element_count),
                self.allocator.sequence_addresses(raw.ts2, element_count),
                self.allocator.sequence_addresses(raw.ts3, element_count),
            ),
        }
    }

    fn allocate_destinations(&mut self, td: Option<usize>, element_count: usize) -> Vec<usize> {
        (0..element_count)
            .map(|index| {
                self.allocator
                    .allocate_destination(td, element_count, index)
            })
            .collect()
    }

    fn broadcast_addresses(&self, raw: &AtlasInstr, element_count: usize) -> Vec<usize> {
        let source_td = raw
            .ts1
            .expect("Broadcast instructions must provide a ts1 operand");
        let operand_instr = self
            .td_lookup
            .get(&source_td)
            .unwrap_or_else(|| panic!("Missing broadcast operand instruction for td {source_td}"));

        let operand_count = operand_instr.num_output_elements();
        let operand_addresses = self
            .allocator
            .sequence_addresses(operand_instr.td, operand_count);

        let operand_tensor = Tensor::new(
            Some(
                &operand_addresses
                    .iter()
                    .map(|&addr| addr as i32)
                    .collect::<Vec<i32>>(),
            ),
            &operand_instr.output_dims,
        )
        .expect("Operand tensor shape should match recorded output dims");

        let expanded = operand_tensor
            .expand(&raw.output_dims)
            .expect("Broadcast expansion should succeed");

        let broadcasted = expanded
            .data()
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();

        assert_eq!(
            broadcasted.len(),
            element_count,
            "Broadcast expansion must create one address per active output element",
        );

        broadcasted
    }
}

/// Keeps track of the virtual tensor address space while the bytecode is
/// unrolled into scalar instructions. Each `(tensor, sequence_remaining)` pair
/// resolves to a unique virtual address in the Jolt VM memory model. The allocator
/// reserves a contiguous block of addresses for the zero register so it can safely
/// stand in for absent operands during tensor decomposition.
struct VirtualTensorAllocator {
    next_address: usize,
    map: BTreeMap<(usize, usize), usize>,
}

impl VirtualTensorAllocator {
    fn new(zero_register_span: usize) -> Self {
        let mut map = BTreeMap::new();
        let mut next_address = 0;

        for sequence_remaining in (0..zero_register_span).rev() {
            map.insert((0, sequence_remaining), next_address);
            next_address += 1;
        }

        Self { next_address, map }
    }

    fn sequence_addresses(&self, tensor: Option<usize>, length: usize) -> Vec<usize> {
        (0..length)
            .map(|index| {
                let key = (
                    zkvm_address(tensor),
                    tensor_sequence_remaining(length, index),
                );
                *self.map.get(&key).unwrap_or_else(|| {
                    panic!(
                        "Missing virtual address for tensor {key:?}; ensure instructions are ordered correctly"
                    )
                })
            })
            .collect()
    }

    fn allocate_destination(
        &mut self,
        tensor: Option<usize>,
        length: usize,
        index: usize,
    ) -> usize {
        if tensor.is_none() {
            return index;
        }

        let key = (
            zkvm_address(tensor),
            tensor_sequence_remaining(length, index),
        );
        let assigned_address = self.next_address;
        let previous = self.map.insert(key, assigned_address);
        assert!(
            previous.is_none(),
            "Virtual tensor address reassigned for key {key:?}",
        );
        self.next_address += 1;
        assigned_address
    }

    fn finalize(self) -> (BTreeMap<(usize, usize), usize>, usize) {
        (self.map, self.next_address)
    }
}

/// Returns the maximum number of active output elements across all instructions in the bytecode.
/// This is used to determine the size of the zero register space that needs to be reserved in memory.
pub fn max_output_elements(bytecode: &[AtlasInstr]) -> usize {
    bytecode
        .iter()
        .map(|instr| instr.num_output_elements())
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
