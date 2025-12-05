use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    lookup_table::{EqualTable, LookupTables},
};
use onnx_tracer::instructions::virtuals::VirtualAssertEq;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertEq {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(EqualTable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for VirtualAssertEq {
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
        (x == y as u64).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::materialize_entry_test;
    use onnx_tracer::trace_types::AtlasOpcode;

    #[test]
    fn materialize_entry() {
        materialize_entry_test(AtlasOpcode::VirtualAssertEq);
    }
}
