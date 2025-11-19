use crate::jolt::{
    executor::instructions::InstructionLookup,
    lookup_table::{LookupTables, VirtualSRATable},
};
use jolt_core::{utils::lookup_bits::LookupBits, zkvm::instruction::LookupQuery};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualSraInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualSraInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRATable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for VirtualSraInstruction<WORD_SIZE> {
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
        let mut x = LookupBits::new(x, WORD_SIZE);
        let mut y = LookupBits::new(y as u64, WORD_SIZE);

        let sign_bit = if x.leading_ones() == 0 { 0 } else { 1 };
        let mut entry = 0;
        let mut sign_extension = 0;
        for i in 0..WORD_SIZE {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry *= 1 + y_i;
            entry += x_i * y_i;
            if i != 0 {
                sign_extension += (1 << i) * (1 - y_i);
            }
        }
        entry + sign_bit * sign_extension
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::materialize_entry_test;
    use onnx_tracer::trace_types::ONNXOpcode;

    #[test]
    fn materialize_entry() {
        materialize_entry_test(ONNXOpcode::VirtualSra);
    }
}
