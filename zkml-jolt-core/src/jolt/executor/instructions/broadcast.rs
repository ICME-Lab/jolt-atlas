use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    lookup_table::{LookupTables, RangeCheckTable},
};
use onnx_tracer::instructions::broadcast::Broadcast;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for Broadcast {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for Broadcast {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8 as u64, 0),
            32 => (self.0 as u32 as u64, 0),
            64 => (self.0, 0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x + y as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8).into(),
            32 => (self.0 as u32).into(),
            64 => self.0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
