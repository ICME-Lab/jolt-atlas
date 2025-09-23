use crate::jolt::{
    executor::instructions::AtlasInstructionLookup,
    lookup_table::{AtlasLookupTables, jolt_tables::RangeCheckTable},
};
use jolt_core::zkvm::instruction::LookupQuery;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> AtlasInstructionLookup<WORD_SIZE> for MulInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<AtlasLookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for MulInstruction<WORD_SIZE> {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x * y as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8 as u64, self.1 as u8 as i64),
            32 => (self.0 as u32 as u64, self.1 as u32 as i64),
            64 => (self.0, self.1 as i64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as i8).wrapping_mul(y as i8) as u8 as u64,
            32 => (x as i32).wrapping_mul(y as i32) as u32 as u64,
            64 => (x as i64).wrapping_mul(y) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::materialize_entry_test;
    use onnx_tracer::trace_types::ONNXOpcode;
    #[test]
    fn materialize_entry() {
        materialize_entry_test(ONNXOpcode::Mul);
    }
}
