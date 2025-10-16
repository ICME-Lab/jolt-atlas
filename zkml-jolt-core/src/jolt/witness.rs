#![allow(static_mut_refs)]

use itertools::Itertools;
use rayon::prelude::*;

use std::{
    cell::{OnceCell, UnsafeCell},
    collections::HashMap,
    sync::Arc,
};

use crate::jolt::bytecode::CircuitFlags;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial,
    },
    utils::math::Math,
    zkvm::instruction::LookupQuery,
};
use rayon::iter::ParallelIterator;

use crate::jolt::{JoltProverPreprocessing, trace::JoltONNXCycle};

struct SharedWitnessData(UnsafeCell<WitnessData>);
unsafe impl Sync for SharedWitnessData {}

/// K^{1/d}
pub const DTH_ROOT_OF_K: usize = 1 << 8;

pub fn compute_d_parameter_from_log_K(log_K: usize) -> usize {
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

pub fn compute_d_parameter(K: usize) -> usize {
    // Calculate D dynamically such that 2^8 = K^(1/D)
    let log_K = K.log_2();
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum CommittedPolynomial {
    /* R1CS aux variables */
    /// The "left" input to the current instruction. Typically either the
    /// rs1 value or the current program counter.
    LeftInstructionInput,
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput,
    /// Product of `LeftInstructionInput` and `RightInstructionInput`
    Product,
    /// Whether the current instruction should write the lookup output to
    /// the destination register
    WriteLookupOutputToTD,
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    TdInc,
    TdIncS, // HACK(Forpee): I am sure there is a better way to do this (compared to TdInc when we compute it we convert to i32 and back to i64)
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// Product of Ts1Value and OpFlags(CircuitFlags::Select)
    SelectCond,
    /// Product of TdWriteValue and OpFlags(CircuitFlags::Select)
    SelectRes,
}

pub static mut ALL_COMMITTED_POLYNOMIALS: OnceCell<Vec<CommittedPolynomial>> = OnceCell::new();

struct WitnessData {
    // Simple polynomial coefficients
    left_instruction_input: Vec<u64>,
    right_instruction_input: Vec<i64>,
    product: Vec<u64>,
    write_lookup_output_to_td: Vec<u64>,
    select_cond: Vec<u8>,
    select_res: Vec<u64>,
    td_inc: Vec<i64>,
    td_inc_s: Vec<i64>,

    // One-hot polynomial indices
    instruction_ra: [Vec<Option<usize>>; 8],
    bytecode_ra: Vec<Vec<Option<usize>>>,
}

unsafe impl Send for WitnessData {}
unsafe impl Sync for WitnessData {}

impl WitnessData {
    fn new(trace_len: usize, bytecode_d: usize) -> Self {
        Self {
            left_instruction_input: vec![0; trace_len],
            right_instruction_input: vec![0; trace_len],
            product: vec![0; trace_len],
            write_lookup_output_to_td: vec![0; trace_len],
            td_inc: vec![0; trace_len],
            td_inc_s: vec![0; trace_len],
            select_cond: vec![0; trace_len],
            select_res: vec![0; trace_len],
            instruction_ra: [
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
                vec![None; trace_len],
            ],
            bytecode_ra: (0..bytecode_d).map(|_| vec![None; trace_len]).collect(),
        }
    }
}

pub struct AllCommittedPolynomials();
impl AllCommittedPolynomials {
    pub fn initialize(bytecode_d: usize) -> Self {
        let mut polynomials = vec![
            CommittedPolynomial::LeftInstructionInput,
            CommittedPolynomial::RightInstructionInput,
            CommittedPolynomial::Product,
            CommittedPolynomial::SelectCond,
            CommittedPolynomial::SelectRes,
            CommittedPolynomial::WriteLookupOutputToTD,
            CommittedPolynomial::TdInc,
            CommittedPolynomial::TdIncS,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(1),
            CommittedPolynomial::InstructionRa(2),
            CommittedPolynomial::InstructionRa(3),
            CommittedPolynomial::InstructionRa(4),
            CommittedPolynomial::InstructionRa(5),
            CommittedPolynomial::InstructionRa(6),
            CommittedPolynomial::InstructionRa(7),
        ];
        for i in 0..bytecode_d {
            polynomials.push(CommittedPolynomial::BytecodeRa(i));
        }

        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .set(polynomials)
                .expect("ALL_COMMITTED_POLYNOMIALS is already initialized");
        }

        AllCommittedPolynomials()
    }

    pub fn iter() -> impl Iterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
        }
    }

    pub fn par_iter() -> impl ParallelIterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .par_iter()
        }
    }
}

impl Drop for AllCommittedPolynomials {
    fn drop(&mut self) {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .take()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized");
        }
    }
}

impl CommittedPolynomial {
    pub fn len() -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .len()
        }
    }

    // TODO(moodlezoup): return Result<Self>
    pub fn from_index(index: usize) -> Self {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")[index]
        }
    }

    // TODO(moodlezoup): return Result<usize>
    pub fn to_index(&self) -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
                .find_position(|poly| *poly == self)
                .unwrap()
                .0
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness_batch")]
    pub fn generate_witness_batch<F, PCS>(
        polynomials: &[CommittedPolynomial],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[JoltONNXCycle],
    ) -> std::collections::HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        let mut bytecode_d = 0;

        for poly in polynomials {
            if let CommittedPolynomial::BytecodeRa(i) = poly {
                bytecode_d = bytecode_d.max(*i + 1);
            }
        }
        let batch = WitnessData::new(trace.len(), bytecode_d);

        // Precompute constants per cycle
        let bytecode_constants = if bytecode_d > 0 {
            let d = preprocessing.shared.bytecode.d;
            let log_K = preprocessing.shared.bytecode.code_size.log_2();
            let log_K_chunk = log_K.div_ceil(d);
            let K_chunk = 1 << log_K_chunk;
            Some((d, log_K_chunk, K_chunk))
        } else {
            None
        };

        let instruction_ra_shifts: [usize; 8] = std::array::from_fn(|i| {
            jolt_core::zkvm::instruction_lookups::LOG_K_CHUNK
                * (jolt_core::zkvm::instruction_lookups::D - 1 - i)
        });
        let instruction_k_chunk = jolt_core::zkvm::instruction_lookups::K_CHUNK as u64;

        let batch_cell = Arc::new(SharedWitnessData(UnsafeCell::new(batch)));

        // #SAFETY: Each thread writes to a unique index of a pre-allocated vector
        (0..trace.len()).into_par_iter().for_each({
            let batch_cell = batch_cell.clone();
            move |i| {
                let bytecode_line = &preprocessing.shared.bytecode.bytecode[i];
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };
                let (left, right) = LookupQuery::<32>::to_instruction_inputs(cycle);
                let circuit_flags = bytecode_line.circuit_flags();
                let (td_write_flag, (pre_td, post_td)) = (bytecode_line.td, cycle.td_write());
                let ts1_val = cycle.ts1_read();

                batch_ref.left_instruction_input[i] = left;
                batch_ref.right_instruction_input[i] = right;
                batch_ref.product[i] = left * right as u64;

                batch_ref.write_lookup_output_to_td[i] = td_write_flag
                    * (circuit_flags[CircuitFlags::WriteLookupOutputToTD as usize] as u8 as u64);
                batch_ref.select_cond[i] =
                    (ts1_val as u8) * (circuit_flags[CircuitFlags::Select as usize] as u8);
                batch_ref.select_res[i] =
                    (post_td) * (circuit_flags[CircuitFlags::Select as usize] as u8 as u64);

                batch_ref.td_inc[i] = post_td as i64 - pre_td as i64;
                batch_ref.td_inc_s[i] = post_td as i32 as i64 - pre_td as i32 as i64;

                // InstructionRa indices
                let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                for j in 0..8 {
                    let k = (lookup_index >> instruction_ra_shifts[j]) % instruction_k_chunk;
                    batch_ref.instruction_ra[j][i] = Some(k as usize);
                }

                // BytecodeRa indices
                if let Some((d, log_K_chunk, K_chunk)) = bytecode_constants {
                    let pc = preprocessing.shared.bytecode.get_pc(i);

                    for j in 0..bytecode_d {
                        let index = (pc >> (log_K_chunk * (d - 1 - j))) % K_chunk;
                        batch_ref.bytecode_ra[j][i] = Some(index);
                    }
                }
            }
        });

        let mut batch = Arc::try_unwrap(batch_cell)
            .ok()
            .expect("Arc should have single owner")
            .0
            .into_inner();

        // We zero-cost move the data back
        let mut results = HashMap::with_capacity(polynomials.len());

        for poly in polynomials {
            match poly {
                CommittedPolynomial::LeftInstructionInput => {
                    let coeffs = std::mem::take(&mut batch.left_instruction_input);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::RightInstructionInput => {
                    let coeffs = std::mem::take(&mut batch.right_instruction_input);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::Product => {
                    let coeffs = std::mem::take(&mut batch.product);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::WriteLookupOutputToTD => {
                    let coeffs = std::mem::take(&mut batch.write_lookup_output_to_td);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::SelectCond => {
                    let coeffs = std::mem::take(&mut batch.select_cond);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::SelectRes => {
                    let coeffs = std::mem::take(&mut batch.select_res);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::TdInc => {
                    let coeffs = std::mem::take(&mut batch.td_inc);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::TdIncS => {
                    let coeffs = std::mem::take(&mut batch.td_inc_s);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }

                CommittedPolynomial::InstructionRa(i) => {
                    if *i < 8 {
                        let indices = std::mem::take(&mut batch.instruction_ra[*i]);
                        let one_hot = OneHotPolynomial::from_indices(
                            indices,
                            jolt_core::zkvm::instruction_lookups::K_CHUNK,
                        );
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::BytecodeRa(i) => {
                    if *i < bytecode_d {
                        let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                        let d = preprocessing.shared.bytecode.d;
                        let log_K = preprocessing.shared.bytecode.code_size.log_2();
                        let log_K_chunk = log_K.div_ceil(d);
                        let K_chunk = 1 << log_K_chunk;
                        let one_hot = OneHotPolynomial::from_indices(indices, K_chunk);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
            }
        }
        results
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F, PCS>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[JoltONNXCycle],
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::Product => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (left_input, right_input) =
                            LookupQuery::<32>::to_instruction_inputs(cycle);
                        left_input * right_input as u64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::WriteLookupOutputToTD => {
                let coeffs: Vec<u64> = preprocessing
                    .bytecode()
                    .par_iter()
                    .map(|instr| {
                        let flag =
                            instr.circuit_flags()[CircuitFlags::WriteLookupOutputToTD as usize];
                        (instr.td) * (flag as u8 as u64)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::SelectCond => {
                let coeffs: Vec<u8> = preprocessing
                    .bytecode()
                    .par_iter()
                    .zip(trace.par_iter())
                    .map(|(instr, cycle)| {
                        let flag = instr.circuit_flags()[CircuitFlags::Select as usize];
                        (cycle.ts1_read() as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::SelectRes => {
                let coeffs: Vec<u64> = preprocessing
                    .bytecode()
                    .par_iter()
                    .zip(trace.par_iter())
                    .map(|(instr, cycle)| {
                        let flag = instr.circuit_flags()[CircuitFlags::Select as usize];
                        (cycle.td_write().1) * (flag as u8 as u64)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::BytecodeRa(i) => {
                let d = preprocessing.shared.bytecode.d;
                let log_K = preprocessing.shared.bytecode.code_size.log_2();
                let log_K_chunk = log_K.div_ceil(d);
                let K_chunk = 1 << log_K_chunk;
                if *i > d {
                    panic!("Invalid index for bytecode ra: {i}");
                }
                let addresses: Vec<_> = (0..preprocessing.bytecode().len())
                    .into_par_iter()
                    .map(|j| {
                        let pc = preprocessing.shared.bytecode.get_pc(j);
                        Some((pc >> (log_K_chunk * (d - 1 - i))) % K_chunk)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, K_chunk))
            }
            CommittedPolynomial::TdInc => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (pre_value, post_value) = cycle.td_write();
                        post_value as i64 - pre_value as i64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::TdIncS => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (pre_value, post_value) = cycle.td_write();
                        post_value as i32 as i64 - pre_value as i32 as i64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                if *i > jolt_core::zkvm::instruction_lookups::D {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (jolt_core::zkvm::instruction_lookups::LOG_K_CHUNK
                                * (jolt_core::zkvm::instruction_lookups::D - 1 - i)))
                            % jolt_core::zkvm::instruction_lookups::K_CHUNK as u64;
                        Some(k as usize)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    jolt_core::zkvm::instruction_lookups::K_CHUNK,
                ))
            }
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum VirtualPolynomial {
    SpartanAz,
    SpartanBz,
    SpartanCz,
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    LeftLookupOperand,
    RightLookupOperand,
    Td,
    Imm,
    Ts1Value,
    Ts2Value,
    Ts3Value,
    TdWriteValue,
    Ts1Ra,
    Ts2Ra,
    Ts3Ra,
    TdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa,
    RegistersVal,
    OpFlags(CircuitFlags),
    LookupTableFlag(usize),
    // matrix multiplication precompile polys
    AVec(usize),
    BMat(usize),
    CRes(usize),
    RaA(usize),
    RaB(usize),
    RaC(usize),
    // reduce sum precompile polys
    ReduceA(usize),
    ReduceRes(usize),
    ReduceRaA(usize),
    ReduceRaRes(usize),
    ValFinal,
}

// pub static ALL_VIRTUAL_POLYNOMIALS: LazyLock<Vec<VirtualPolynomial>> = LazyLock::new(|| {
//     let mut polynomials = vec![
//         VirtualPolynomial::SpartanAz,
//         VirtualPolynomial::SpartanBz,
//         VirtualPolynomial::SpartanCz,
//         VirtualPolynomial::PC,
//         VirtualPolynomial::UnexpandedPC,
//         VirtualPolynomial::NextPC,
//         VirtualPolynomial::NextUnexpandedPC,
//         VirtualPolynomial::NextIsNoop,
//         VirtualPolynomial::LeftLookupOperand,
//         VirtualPolynomial::RightLookupOperand,
//         VirtualPolynomial::Td,
//         VirtualPolynomial::Imm,
//         VirtualPolynomial::Ts1Value,
//         VirtualPolynomial::Ts2Value,
//         VirtualPolynomial::TdWriteValue,
//         VirtualPolynomial::Ts1Ra,
//         VirtualPolynomial::Ts2Ra,
//         VirtualPolynomial::TdWa,
//         VirtualPolynomial::LookupOutput,
//         VirtualPolynomial::InstructionRaf,
//         VirtualPolynomial::InstructionRafFlag,
//         VirtualPolynomial::InstructionRa,
//         VirtualPolynomial::RegistersVal,
//     ];
//     for flag in CircuitFlags::iter() {
//         polynomials.push(VirtualPolynomial::OpFlags(flag));
//     }
//     for table in LookupTables::iter() {
//         polynomials.push(VirtualPolynomial::LookupTableFlag(
//             LookupTables::<32>::enum_index(&table),
//         ));
//     }

//     polynomials
// });

// impl VirtualPolynomial {
//     pub fn from_index(index: usize) -> Self {
//         ALL_VIRTUAL_POLYNOMIALS[index]
//     }

//     pub fn to_index(&self) -> usize {
//         ALL_VIRTUAL_POLYNOMIALS
//             .iter()
//             .find_position(|poly| *poly == self)
//             .unwrap()
//             .0
//     }
// }
