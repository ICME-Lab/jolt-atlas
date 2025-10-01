use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::trace_types::ONNXOpcode;
use rand::prelude::*;

use crate::jolt::{executor::instructions::InstructionLookup, trace::JoltONNXCycle};

pub fn materialize_entry_test(opcode: ONNXOpcode) {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let cycle_lookup = JoltONNXCycle::random(opcode.clone(), &mut rng)
            .lookup
            .unwrap();
        let table = cycle_lookup.lookup_table().unwrap();
        assert_eq!(
            cycle_lookup.to_lookup_output(),
            table.materialize_entry(cycle_lookup.to_lookup_index())
        );
    }
}
