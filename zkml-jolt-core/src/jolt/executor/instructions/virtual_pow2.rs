use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    lookup_table::{LookupTables, Pow2Table},
};
use onnx_tracer::instructions::virtuals::VirtualPow2;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualPow2 {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(Pow2Table.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for VirtualPow2 {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x + y as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8 as u64, 0),
            32 => (self.0 as u32 as u64, 0),
            64 => (self.0, 0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<WORD_SIZE>::to_lookup_index(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => 1u64 << (y % 8),
            32 => 1u64 << (y % 32),
            64 => 1u64 << (y % 64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::materialize_entry_test;
    use onnx_tracer::trace_types::AtlasOpcode;

    #[test]
    fn materialize_entry() {
        materialize_entry_test(AtlasOpcode::VirtualPow2);
    }
}
