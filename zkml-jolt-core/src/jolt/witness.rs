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
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    utils::math::Math,
    zkvm::instruction::LookupQuery,
};
use rayon::iter::ParallelIterator;

use crate::jolt::trace::JoltONNXCycle;

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
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    TdInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
}

pub static mut ALL_COMMITTED_POLYNOMIALS: OnceCell<Vec<CommittedPolynomial>> = OnceCell::new();

struct WitnessData {
    // Simple polynomial coefficients
    td_inc: Vec<i64>,

    // One-hot polynomial indices
    instruction_ra: [Vec<Option<usize>>; 8],
}

unsafe impl Send for WitnessData {}
unsafe impl Sync for WitnessData {}

impl WitnessData {
    fn new(trace_len: usize) -> Self {
        Self {
            td_inc: vec![0; trace_len],
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
        }
    }
}

pub struct AllCommittedPolynomials();
impl AllCommittedPolynomials {
    pub fn initialize() -> Self {
        let polynomials = vec![
            CommittedPolynomial::TdInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(1),
            CommittedPolynomial::InstructionRa(2),
            CommittedPolynomial::InstructionRa(3),
            CommittedPolynomial::InstructionRa(4),
            CommittedPolynomial::InstructionRa(5),
            CommittedPolynomial::InstructionRa(6),
            CommittedPolynomial::InstructionRa(7),
        ];

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
    pub fn generate_witness_batch<F>(
        polynomials: &[CommittedPolynomial],
        trace: &[JoltONNXCycle],
    ) -> std::collections::HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
    where
        F: JoltField,
    {
        let batch = WitnessData::new(trace.len());

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
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };
                let (pre_td, post_td) = cycle.td_write();
                batch_ref.td_inc[i] = post_td as i64 - pre_td as i64;

                // InstructionRa indices
                let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                for j in 0..8 {
                    let k = (lookup_index >> instruction_ra_shifts[j]) % instruction_k_chunk;
                    batch_ref.instruction_ra[j][i] = Some(k as usize);
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
                CommittedPolynomial::TdInc => {
                    let coeffs = std::mem::take(&mut batch.td_inc);
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
            }
        }
        results
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(&self, trace: &[JoltONNXCycle]) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
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
    NextPC,
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
    Product,
    WriteLookupOutputToTD,
    SelectCond,
    SelectRes,
    LeftInstructionInput,
    RightInstructionInput,
    // precompile polys
    PrecompileA(usize),
    PrecompileB(usize),
    PrecompileC(usize),
    RaAPrecompile(usize),
    RaBPrecompile(usize),
    RaCPrecompile(usize),
    ValFinal,

    // HACK(Forpee): Temporary virtual polynomial for testing
    TdIncS,
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
