use crate::jolt::bytecode::CircuitFlags;
use jolt_core::field::JoltField;

use crate::jolt::r1cs::{
    builder::{CombinedUniformBuilder, R1CSBuilder},
    inputs::JoltONNXR1CSInputs,
};

pub trait R1CSConstraints<F: JoltField> {
    fn construct_constraints(padded_trace_length: usize) -> CombinedUniformBuilder<F> {
        let mut uniform_builder = R1CSBuilder::new();
        Self::uniform_constraints(&mut uniform_builder);

        CombinedUniformBuilder::construct(uniform_builder, padded_trace_length)
    }
    /// Constructs Jolt's uniform constraints.
    /// Uniform constraints are constraints that hold for each step of
    /// the execution trace.
    fn uniform_constraints(builder: &mut R1CSBuilder);
}

pub struct JoltONNXConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltONNXConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder) {
        // if LeftOperandIsRs1Value { assert!(LeftInstructionInput == Ts1Value) }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsTs1Value),
            JoltONNXR1CSInputs::LeftInstructionInput,
            JoltONNXR1CSInputs::Ts1Value,
        );

        // if !(LeftOperandIsTs1Value)  {
        //     assert!(LeftInstructionInput == 0)
        // }
        cs.constrain_eq_conditional(
            1 - JoltONNXR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsTs1Value),
            JoltONNXR1CSInputs::LeftInstructionInput,
            0,
        );

        // if RightOperandIsTs2Value { assert!(RightInstructionInput == Ts2Value) }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsTs2Value),
            JoltONNXR1CSInputs::RightInstructionInput,
            JoltONNXR1CSInputs::Ts2Value,
        );

        // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltONNXR1CSInputs::RightInstructionInput,
            JoltONNXR1CSInputs::Imm,
        );

        // if !(RightOperandIsTs2Value || RightOperandIsImm)  {
        //     assert!(RightInstructionInput == 0)
        // }
        // Note that RightOperandIsTs2Value and RightOperandIsImm are mutually exclusive flags
        cs.constrain_eq_conditional(
            1 - JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsTs2Value)
                - JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltONNXR1CSInputs::RightInstructionInput,
            0,
        );

        // if AddOperands || SubtractOperands || MultiplyOperands {
        //     // Lookup query is just RightLookupOperand
        //     assert!(LeftLookupOperand == 0)
        // } else {
        //     assert!(LeftLookupOperand == LeftInstructionInput)
        // }
        cs.constrain_if_else(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                + JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                + JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            0,
            JoltONNXR1CSInputs::LeftInstructionInput,
            JoltONNXR1CSInputs::LeftLookupOperand,
        );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltONNXR1CSInputs::RightLookupOperand,
            JoltONNXR1CSInputs::LeftInstructionInput + JoltONNXR1CSInputs::RightInstructionInput,
        );

        // If SubtractOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
            JoltONNXR1CSInputs::RightLookupOperand,
            // Converts from unsigned to twos-complement representation
            JoltONNXR1CSInputs::LeftInstructionInput - JoltONNXR1CSInputs::RightInstructionInput
                + (0xffffffffi64 + 1),
        );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
        // }
        cs.constrain_prod(
            JoltONNXR1CSInputs::RightInstructionInput,
            JoltONNXR1CSInputs::LeftInstructionInput,
            JoltONNXR1CSInputs::Product,
        );
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltONNXR1CSInputs::RightLookupOperand,
            JoltONNXR1CSInputs::Product,
        );

        // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
        //     assert!(RightLookupOperand == RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            1 - JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                - JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                - JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands)
                - JoltONNXR1CSInputs::OpFlags(CircuitFlags::Const)
                // Arbitrary untrusted advice goes in right lookup operand
                - JoltONNXR1CSInputs::OpFlags(CircuitFlags::Advice),
            JoltONNXR1CSInputs::RightLookupOperand,
            JoltONNXR1CSInputs::RightInstructionInput,
        );

        // if Assert {
        //     assert!(LookupOutput == 1)
        // }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::Assert),
            JoltONNXR1CSInputs::LookupOutput,
            1,
        );

        // if Select && Condition (Ts1Value) {
        //     assert!(TdWriteValue == Ts2Value)
        // } else if Select && !Condition (Ts1Value) {
        //     assert!(TdWriteValue == Ts3Value)
        // } else {
        //     assert!(TdWriteValue == /* Further assertions down below */)
        // }
        cs.constrain_prod(
            JoltONNXR1CSInputs::Ts1Value,
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::Select),
            JoltONNXR1CSInputs::SelectCond,
        );
        cs.constrain_prod(
            JoltONNXR1CSInputs::TdWriteValue,
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::Select),
            JoltONNXR1CSInputs::SelectRes,
        );
        cs.constrain_if_else(
            JoltONNXR1CSInputs::SelectCond,
            JoltONNXR1CSInputs::Ts2Value,
            JoltONNXR1CSInputs::Ts3Value,
            JoltONNXR1CSInputs::SelectRes,
        );

        // if !Select {
        //     assert!(Ts3Value == 0)
        // }
        cs.constrain_eq_conditional(
            1 - JoltONNXR1CSInputs::OpFlags(CircuitFlags::Select),
            JoltONNXR1CSInputs::Ts3Value,
            0,
        );

        // if Td != 0 && WriteLookupOutputToTD {
        //     assert!(TdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltONNXR1CSInputs::Td,
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToTD),
            JoltONNXR1CSInputs::WriteLookupOutputToTD,
        );
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::WriteLookupOutputToTD,
            JoltONNXR1CSInputs::TdWriteValue,
            JoltONNXR1CSInputs::LookupOutput,
        );

        // if CircuitFlag::Const {
        //     assert!(TdWriteValue == Const)
        // }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::Const),
            JoltONNXR1CSInputs::Imm,
            JoltONNXR1CSInputs::TdWriteValue,
        );

        // If Halt {
        //     assert!(NextPC == PC + 1)
        // } else {
        //    assert!(NextPC == 0)
        // }
        cs.constrain_if_else(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::Halt),
            0,
            JoltONNXR1CSInputs::PC + 1,
            JoltONNXR1CSInputs::NextPC,
        );
    }
}
