#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use std::marker::PhantomData;

use crate::jolt::{
    JoltProverPreprocessing,
    bytecode::CircuitFlags,
    pcs::{OpeningId, SumcheckId},
    r1cs::{key::UniformSpartanKey, spartan::UniformSpartanProof},
    trace::JoltONNXCycle,
    witness::VirtualPolynomial,
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
    zkvm::{
        instruction::LookupQuery,
        r1cs::ops::{LC, Term, Variable},
    },
};
use rayon::prelude::*;

pub struct R1CSProof<F: JoltField, ProofTranscript: Transcript> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, ProofTranscript>,
    pub _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltONNXR1CSInputs {
    PC,                    // Virtual (bytecode raf)
    Td,                    // Virtual (bytecode rv)
    Imm,                   // Virtual (bytecode rv)
    Ts1Value,              // Virtual (registers rv)
    Ts2Value,              // Virtual (registers rv)
    Ts3Value,              // Virtual (registers rv)
    TdWriteValue,          // Virtual (registers wv)
    LeftInstructionInput,  // to_lookup_query -> to_instruction_operands
    RightInstructionInput, // to_lookup_query -> to_instruction_operands
    LeftLookupOperand,     // Virtual (instruction raf)
    RightLookupOperand,    // Virtual (instruction raf)
    Product,               // LeftInstructionOperand * RightInstructionOperand
    WriteLookupOutputToTD,
    NextPC,       // Virtual (spartan shift sumcheck)
    LookupOutput, // Virtual (instruction rv)
    SelectCond,   // Ts1Value * Select
    SelectRes,    // TdWriteValue * Select
    OpFlags(CircuitFlags),
}

impl TryFrom<JoltONNXR1CSInputs> for VirtualPolynomial {
    type Error = &'static str;

    fn try_from(value: JoltONNXR1CSInputs) -> Result<Self, Self::Error> {
        match value {
            JoltONNXR1CSInputs::PC => Ok(VirtualPolynomial::PC),
            JoltONNXR1CSInputs::Td => Ok(VirtualPolynomial::Td),
            JoltONNXR1CSInputs::Imm => Ok(VirtualPolynomial::Imm),
            JoltONNXR1CSInputs::Ts1Value => Ok(VirtualPolynomial::Ts1Value),
            JoltONNXR1CSInputs::Ts2Value => Ok(VirtualPolynomial::Ts2Value),
            JoltONNXR1CSInputs::Ts3Value => Ok(VirtualPolynomial::Ts3Value),
            JoltONNXR1CSInputs::TdWriteValue => Ok(VirtualPolynomial::TdWriteValue),
            JoltONNXR1CSInputs::LeftLookupOperand => Ok(VirtualPolynomial::LeftLookupOperand),
            JoltONNXR1CSInputs::RightLookupOperand => Ok(VirtualPolynomial::RightLookupOperand),
            JoltONNXR1CSInputs::NextPC => Ok(VirtualPolynomial::NextPC),
            JoltONNXR1CSInputs::LookupOutput => Ok(VirtualPolynomial::LookupOutput),
            JoltONNXR1CSInputs::OpFlags(flag) => Ok(VirtualPolynomial::OpFlags(flag)),
            JoltONNXR1CSInputs::LeftInstructionInput => Ok(VirtualPolynomial::LeftInstructionInput),
            JoltONNXR1CSInputs::RightInstructionInput => {
                Ok(VirtualPolynomial::RightInstructionInput)
            }
            JoltONNXR1CSInputs::Product => Ok(VirtualPolynomial::Product),
            JoltONNXR1CSInputs::SelectCond => Ok(VirtualPolynomial::SelectCond),
            JoltONNXR1CSInputs::SelectRes => Ok(VirtualPolynomial::SelectRes),
            JoltONNXR1CSInputs::WriteLookupOutputToTD => {
                Ok(VirtualPolynomial::WriteLookupOutputToTD)
            }
        }
    }
}

impl TryFrom<JoltONNXR1CSInputs> for OpeningId {
    type Error = &'static str;

    fn try_from(value: JoltONNXR1CSInputs) -> Result<Self, Self::Error> {
        if let Ok(poly) = VirtualPolynomial::try_from(value) {
            Ok(OpeningId::Virtual(poly, SumcheckId::SpartanOuter))
        } else {
            Err("Could not map {value} to an OpeningId")
        }
    }
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltONNXR1CSInputs; 29] = [
    JoltONNXR1CSInputs::LeftInstructionInput,
    JoltONNXR1CSInputs::RightInstructionInput,
    JoltONNXR1CSInputs::Product,
    JoltONNXR1CSInputs::WriteLookupOutputToTD,
    JoltONNXR1CSInputs::PC,
    JoltONNXR1CSInputs::Td,
    JoltONNXR1CSInputs::Imm,
    JoltONNXR1CSInputs::Ts1Value,
    JoltONNXR1CSInputs::Ts2Value,
    JoltONNXR1CSInputs::Ts3Value,
    JoltONNXR1CSInputs::TdWriteValue,
    JoltONNXR1CSInputs::LeftLookupOperand,
    JoltONNXR1CSInputs::RightLookupOperand,
    JoltONNXR1CSInputs::NextPC,
    JoltONNXR1CSInputs::LookupOutput,
    JoltONNXR1CSInputs::SelectCond,
    JoltONNXR1CSInputs::SelectRes,
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsTs1Value),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsTs2Value),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToTD),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::Advice),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::Const),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::Select),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::Halt),
];

impl JoltONNXR1CSInputs {
    /// The total number of unique constraint inputs
    pub fn num_inputs() -> usize {
        ALL_R1CS_INPUTS.len()
    }

    /// Converts an index to the corresponding constraint input.
    pub fn from_index(index: usize) -> Self {
        ALL_R1CS_INPUTS[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ALL_R1CS_INPUTS`.
    pub fn to_index(&self) -> usize {
        match ALL_R1CS_INPUTS.iter().position(|x| x == self) {
            Some(index) => index,
            None => panic!("Invalid variant {self:?}"),
        }
    }

    pub fn generate_witness<F, PCS>(
        &self,
        trace: &[JoltONNXCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            JoltONNXR1CSInputs::PC => {
                let coeffs: Vec<u64> = (0..preprocessing.bytecode().len())
                    .into_par_iter()
                    .map(|i| preprocessing.shared.bytecode.get_pc(i) as u64)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::NextPC => {
                let coeffs: Vec<u64> = (0..preprocessing.bytecode().len())
                    .into_par_iter()
                    .skip(1)
                    .map(|i| preprocessing.shared.bytecode.get_pc(i) as u64)
                    .chain(rayon::iter::once(0))
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Td => {
                let coeffs: Vec<u64> = preprocessing
                    .bytecode()
                    .par_iter()
                    .map(|instr| instr.td)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Imm => {
                let coeffs: Vec<u64> = preprocessing
                    .bytecode()
                    .par_iter()
                    .map(|instr| instr.imm)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Ts1Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.ts1_read()).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Ts2Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.ts2_read()).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Ts3Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.ts3_read()).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::TdWriteValue => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.td_write().1).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::LeftLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).0)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RightLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).1)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Product => {
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
            JoltONNXR1CSInputs::SelectCond => {
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
            JoltONNXR1CSInputs::SelectRes => {
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
            JoltONNXR1CSInputs::WriteLookupOutputToTD => {
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
            JoltONNXR1CSInputs::LookupOutput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(LookupQuery::<32>::to_lookup_output)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::OpFlags(flag) => {
                let coeffs: Vec<u8> = preprocessing
                    .bytecode()
                    .par_iter()
                    .map(|instr| instr.circuit_flags()[*flag as usize] as u8)
                    .collect();
                coeffs.into()
            }
        }
    }
}

impl From<JoltONNXR1CSInputs> for Variable {
    fn from(input: JoltONNXR1CSInputs) -> Variable {
        Variable::Input(input.to_index())
    }
}

impl From<JoltONNXR1CSInputs> for Term {
    fn from(input: JoltONNXR1CSInputs) -> Term {
        Term(Variable::Input(input.to_index()), 1)
    }
}

impl From<JoltONNXR1CSInputs> for LC {
    fn from(input: JoltONNXR1CSInputs) -> LC {
        Term(Variable::Input(input.to_index()), 1).into()
    }
}

/// Newtype wrapper to allow conversion from a vector of inputs to LC.
pub struct InputVec(pub Vec<JoltONNXR1CSInputs>);

impl From<InputVec> for LC {
    fn from(input_vec: InputVec) -> LC {
        let terms: Vec<Term> = input_vec.0.into_iter().map(Into::into).collect();
        LC::new(terms)
    }
}

impl<T: Into<LC>> std::ops::Add<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn add(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc + rhs_lc
    }
}
impl<T: Into<LC>> std::ops::Sub<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn sub(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc - rhs_lc
    }
}
impl std::ops::Mul<i64> for JoltONNXR1CSInputs {
    type Output = Term;
    fn mul(self, rhs: i64) -> Self::Output {
        Term(Variable::Input(self.to_index()), rhs)
    }
}
impl std::ops::Mul<JoltONNXR1CSInputs> for i64 {
    type Output = Term;
    fn mul(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        Term(Variable::Input(rhs.to_index()), self)
    }
}
impl std::ops::Add<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn add(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), 1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}
impl std::ops::Sub<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn sub(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), -1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_to_index() {
        for i in 0..JoltONNXR1CSInputs::num_inputs() {
            assert_eq!(i, JoltONNXR1CSInputs::from_index(i).to_index());
        }
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var,
                JoltONNXR1CSInputs::from_index(JoltONNXR1CSInputs::to_index(&var))
            );
        }
    }
}
