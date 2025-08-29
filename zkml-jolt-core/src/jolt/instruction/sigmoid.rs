use crate::jolt::execution_trace::ONNXLookupQuery;
use jolt_core::jolt::{instruction::{InstructionLookup, LookupQuery}, lookup_table::{sigmoid::{SCALE, SigmoidTable}, LookupTables}};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use serde::{Deserialize, Serialize};


pub fn compute_sigmoid(x: u64) -> u64 {
    let x_dequant = x as f32 / SCALE;
    let sigmoid = 1.0 / (1.0 + (-x_dequant).exp());
    let x_requant = (sigmoid * SCALE).round();
    x_requant as u64
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for SigmoidInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(SigmoidTable::<WORD_SIZE>.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for SigmoidInstruction<WORD_SIZE> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (self.0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, _) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (x, 0)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }


    fn to_lookup_output(&self) -> u64 {
        let (x, _) = self.to_instruction_inputs();
        compute_sigmoid(x)
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::execution_trace::JoltONNXCycle;

    use super::*;
    use jolt_core::jolt::instruction::InstructionLookup;
    use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]  
    fn materialize_entry() {
        let mut rng = StdRng::seed_from_u64(12345);
        let cycle: ONNXCycle = ONNXCycle::random(ONNXOpcode::Sigmoid, &mut rng);
        let jolt_cycle = JoltONNXCycle::from_raw(&cycle);
        println!("jolt_cycle: {:?}", jolt_cycle);
        let table = jolt_cycle.lookup_table().unwrap();
        for _ in 0..1 {
            // let random_cycle = jolt_cycle.random(&mut rng);
            let output = <JoltONNXCycle as ONNXLookupQuery<32>>::to_lookup_output(&jolt_cycle);
            println!("output: {:?}", output[0]);
            let index = <JoltONNXCycle as ONNXLookupQuery<32>>::to_lookup_index(&jolt_cycle);
            println!("index: {:?}", index[0]);
            let materialized_entry = table.materialize_entry(index[0]);
            println!("materialized_entry: {:?}", materialized_entry);
            assert_eq!(
                output[0],
                materialized_entry,
                "{:?}",
                jolt_cycle.memory_ops
            );
        }
    }
}
