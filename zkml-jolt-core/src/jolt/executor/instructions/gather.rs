use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};
use onnx_tracer::instructions::gather::Gather;

// This instruction is not handled via lookup tables, but rather with specialized sum-check.
impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for Gather {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for Gather {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        (0, 0)
    }

    fn to_lookup_index(&self) -> u64 {
        0
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
