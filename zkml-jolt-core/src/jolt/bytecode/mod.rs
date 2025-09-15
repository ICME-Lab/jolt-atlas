use crate::jolt::{
    bytecode::{
        booleanity::BooleanitySumcheck, hamming_weight::HammingWeightSumcheck,
        read_raf_checking::ReadRafSumcheck,
    },
    dag::{stage::SumcheckStages, state_manager::StateManager},
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
        instruction::InstructionLookup,
        lookup_table::{LookupTables, range_check::RangeCheckTable},
        witness::{DTH_ROOT_OF_K, compute_d_parameter},
    },
};
use onnx_tracer::{
    graph::model::Model,
    trace_types::{CircuitFlags, NUM_CIRCUIT_FLAGS, ONNXInstr, ONNXOpcode},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// Key: (ELF address, virtual sequence index or 0)
    pub tensor_virtual_pc_map: BTreeMap<(usize, usize), usize>,
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
    /// "Immediate" value for this instruction (0 if unused).
    pub imm: u64,
    /// Element-wise operations are decomposed into sequences of scalar instructions during preprocessing.
    /// This field tracks the remaining operations in such sequences for proper execution order.
    pub tensor_sequence_remaining: Option<usize>,
}

impl JoltONNXBytecode {
    pub fn no_op() -> Self {
        Self {
            address: 0,
            opcode: ONNXOpcode::Noop,
            td: 0,
            ts1: 0,
            ts2: 0,
            imm: 0,
            tensor_sequence_remaining: None,
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
            | ONNXOpcode::Sum
            | ONNXOpcode::Relu
            | ONNXOpcode::Output
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
            | ONNXOpcode::Sum
            | ONNXOpcode::Relu
            | ONNXOpcode::Output
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

        // flags[CircuitFlags::Precompile as usize] = matches!(
        //     self.opcode,
        //     ONNXOpcode::MatMult
        // );
        flags[CircuitFlags::Assert as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
        );

        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.tensor_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.tensor_sequence_remaining.unwrap_or(0) != 0;

        // // TODO(Forpee): These single-opcode flags could be simplified to direct equality checks
        // // unlike the multi-opcode matches above. We could Consider refactoring to use a more
        // // systematic approach like opcode-to-flag mapping or trait-based dispatch.
        // flags[CircuitFlags::SumOperands as usize] = self.opcode == ONNXOpcode::Sum;
        // flags[CircuitFlags::Gather as usize] = self.opcode == ONNXOpcode::Gather;
        // flags[CircuitFlags::Select as usize] = self.opcode == ONNXOpcode::Select;
        // flags[CircuitFlags::BroadCast as usize] = self.opcode == ONNXOpcode::Broadcast;

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
        let (mut bytecode, memory_K) = Self::inline_tensor_instrs(model);
        let mut tensor_virtual_pc_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            assert_eq!(
                tensor_virtual_pc_map.insert(
                    (
                        instruction.address,
                        instruction.tensor_sequence_remaining.unwrap_or(0)
                    ),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, JoltONNXBytecode::no_op());
        assert_eq!(tensor_virtual_pc_map.insert((0, 0), 0), None);
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
        }
    }

    pub fn inline_tensor_instrs<ModelFunc>(model: ModelFunc) -> (Vec<JoltONNXBytecode>, usize)
    where
        ModelFunc: Fn() -> Model,
    {
        let bytecode = onnx_tracer::decode_model(model());
        println!("program bytecode: {bytecode:#?}");
        let mut preprocessed_bytecode: Vec<JoltONNXBytecode> = Vec::new();

        // Memory management and instruction preprocessing:
        // 1. Allocate virtual memory addresses for tensor operands and results
        // 2. Decompose tensor operations into scalar instructions with individual immediate values
        // 3. Map ONNX instruction operands to virtual register addresses for the Jolt VM

        // virtual tensor address
        let mut vt_address: usize = 0;
        // virtual tensor address map
        let mut vt_address_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let max_active_output_elements = max_output_elements(&bytecode);

        // reserve address space for zero registers
        for i in 0..max_active_output_elements {
            vt_address_map.insert((0, max_active_output_elements - i - 1), vt_address);
            vt_address += 1;
        }

        // convert each ONNX instruction to one or more Joule bytecode lines
        // and allocate virtual memory addresses for their operands and results
        // also update the virtual tensor address map
        // to map ONNX tensor addresses to Jolt VM virtual addresses
        // e.g., ONNX tensor address 0 (zero register) maps to virtual addresses 0..max_active_output_elements
        // e.g., ONNX tensor address 1 maps to virtual addresses max_active_output_elements..(max_active_output_elements + its size)
        // etc.
        for instruction in bytecode.into_iter() {
            let joule_instructions =
                raw_to_joule(instruction, &mut vt_address, &mut vt_address_map);
            preprocessed_bytecode.extend(joule_instructions);
        }
        println!("preprocessed bytecode: {preprocessed_bytecode:#?}");
        (preprocessed_bytecode, vt_address.next_power_of_two())
    }

    pub fn get_pc(&self, i: usize) -> usize {
        *self
            .tensor_virtual_pc_map
            .get(&(
                self.bytecode[i].address,
                self.bytecode[i].tensor_sequence_remaining.unwrap_or(0),
            ))
            .unwrap()
    }
}

pub fn raw_to_joule(
    raw: ONNXInstr,
    vt_address: &mut usize,
    vt_address_map: &mut BTreeMap<(usize, usize), usize>,
) -> Vec<JoltONNXBytecode> {
    let mut joule_instructions: Vec<JoltONNXBytecode> = vec![];
    // Address allocation strategy:
    // 1. Reserve memory slots for zero registers (max active output elements across all instructions)
    // 2. For each scalar operation derived from tensor decomposition:
    //    - ts1/ts2: Read from previously allocated addresses (from prior instruction outputs)
    //    - td: Allocate new virtual memory address for this instruction's result
    // 3. Map ONNX tensor addresses to virtual memory addresses for Jolt VM execution

    // TODO(Forpee): Does not work with non elementwise ops (will need to come-back to this for matmult)
    let active_output_elements = raw.active_output_elements;

    // get ts1 and ts2 addresses
    let vts1 = (0..active_output_elements)
        .map(|i| {
            vt_address_map[&(
                zkvm_address(raw.ts1),
                tensor_sequence_remaining(active_output_elements, i),
            )]
        })
        .collect::<Vec<usize>>();
    let vts2 = (0..active_output_elements)
        .map(|i| {
            vt_address_map[&(
                zkvm_address(raw.ts2),
                tensor_sequence_remaining(active_output_elements, i),
            )]
        })
        .collect::<Vec<usize>>();

    // calculate td address
    let vtd = (0..active_output_elements)
        .map(|i| {
            if raw.td.is_some() {
                let addr = *vt_address;
                assert_eq!(
                    vt_address_map.insert(
                        (
                            zkvm_address(raw.td),
                            tensor_sequence_remaining(active_output_elements, i),
                        ),
                        addr,
                    ),
                    None
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
        joule_instructions.push(JoltONNXBytecode {
            address: raw.address,
            opcode: raw.opcode.clone(),
            td: vtd[i] as u64,
            ts1: vts1[i] as u64,
            ts2: vts2[i] as u64,
            imm: imm[i] as u64,
            tensor_sequence_remaining: Some(tensor_sequence_remaining(active_output_elements, i)),
        });
    }
    joule_instructions
}

pub fn max_output_elements(bytecode: &[ONNXInstr]) -> usize {
    bytecode
        .iter()
        .map(|instr| instr.active_output_elements)
        .max()
        .unwrap_or(1)
}

#[inline]
pub fn zkvm_address(t: Option<usize>) -> usize {
    t.map_or(0, |t| t + ZERO_ADDR_PREPEND)
}

#[inline]
pub fn tensor_sequence_remaining(active_output_elements: usize, current_index: usize) -> usize {
    active_output_elements - current_index - 1
}
