use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    trace::JoltONNXCycle,
};
use onnx_tracer::trace_types::AtlasOpcode;
use rand::prelude::*;

pub fn materialize_entry_test(opcode: AtlasOpcode) {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let cycle = JoltONNXCycle::random(opcode.clone(), &mut rng);
        let table = cycle.lookup_table().unwrap();
        assert_eq!(
            cycle.to_lookup_output(),
            table.materialize_entry(cycle.to_lookup_index())
        );
    }
}
