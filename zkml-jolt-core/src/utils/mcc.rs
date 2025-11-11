#[cfg(test)]
use crate::jolt::{bytecode::JoltONNXBytecode, trace::JoltONNXCycle};

/// This function validates that the execution trace matches the expected memory
/// operations by simulating the execution and checking that reads and writes
/// occur at the correct memory addresses with the expected values.
///
/// # Purpose
///
/// This validation is crucial because runtime sometimes converts operands to
/// floating point for intermediate calculations, which can cause mismatches
/// between expected outputs and actual trace values. This function helps catch
/// such discrepancies during testing.
///
/// # Arguments
///
/// * `bytecode` - The sequence of Jolt ONNX bytecode instructions
/// * `execution_trace` - The corresponding execution trace cycles
/// * `memory_size` - The size of the memory space (number of memory addresses)
///
/// # Validation Process
///
/// For each instruction-cycle pair:
/// 1. Validates that ts1 and ts2 reads match the current memory state
/// 2. Validates that td pre-write value matches current memory at td address
/// 3. Updates memory with the td post-write value
///
/// # Panics
///
/// This function will panic with detailed error information if any memory
/// operation doesn't match the expected values, including:
/// - The cycle number where the error occurred
/// - The instruction and cycle details
/// - Expected vs actual values
/// - The memory address involved
///
/// # Example Error Output
///
/// ```text
/// TS1 READ error at cycle_42: <instruction> <cycle>; Expected: 123, got: 456 at address 78
/// ```
#[cfg(test)]
pub fn sanity_check_mcc(
    bytecode: &[JoltONNXBytecode],
    execution_trace: &[JoltONNXCycle],
    memory_size: usize,
) {
    assert_eq!(
        bytecode.len(),
        execution_trace.len(),
        "Bytecode and execution trace must have the same length"
    );

    // Initialize memory with all zeros
    let mut memory = vec![0u64; memory_size];

    // Validate each cycle against its corresponding instruction
    for (cycle_index, (cycle, instr)) in execution_trace.iter().zip(bytecode.iter()).enumerate() {
        validate_memory_operation(cycle_index, cycle, instr, &mut memory);
    }
}

/// Validates a single memory operation against the current memory state.
///
/// # Arguments
///
/// * `cycle_index` - The index of the current cycle (for error reporting)
/// * `cycle` - The execution cycle to validate
/// * `instr` - The corresponding instruction
/// * `memory` - The current memory state (will be updated with td post-write value)
#[cfg(test)]
fn validate_memory_operation(
    cycle_index: usize,
    cycle: &JoltONNXCycle,
    instr: &JoltONNXBytecode,
    memory: &mut [u64],
) {
    // Extract memory addresses from the instruction
    let addresses = MemoryAddresses::from_instruction(instr);

    // Validate reads from source registers
    validate_read(
        "TS1",
        cycle_index,
        addresses.ts1,
        cycle.ts1_read(),
        memory,
        instr,
        cycle,
    );
    validate_read(
        "TS2",
        cycle_index,
        addresses.ts2,
        cycle.ts2_read(),
        memory,
        instr,
        cycle,
    );

    // Validate destination register pre-write state
    let (td_pre, td_post) = cycle.td_write();
    validate_read(
        "TD",
        cycle_index,
        addresses.td,
        td_pre,
        memory,
        instr,
        cycle,
    );

    // Update memory with the post-write value
    memory[addresses.td] = td_post;
}

/// Helper struct to hold memory addresses for cleaner code.
#[cfg(test)]
struct MemoryAddresses {
    ts1: usize,
    ts2: usize,
    td: usize,
}

#[cfg(test)]
impl MemoryAddresses {
    fn from_instruction(instr: &JoltONNXBytecode) -> Self {
        Self {
            ts1: instr.ts1 as usize,
            ts2: instr.ts2 as usize,
            td: instr.td as usize,
        }
    }
}

/// Validates a single read operation.
///
/// # Arguments
///
/// * `register_name` - Name of the register being validated (for error messages)
/// * `cycle_index` - The cycle index (for error reporting)
/// * `address` - The memory address being read
/// * `actual_value` - The value that was read according to the trace
/// * `memory` - The current memory state
/// * `instr` - The instruction (for error reporting)
/// * `cycle` - The cycle (for error reporting)
#[cfg(test)]
fn validate_read(
    register_name: &str,
    cycle_index: usize,
    address: usize,
    actual_value: u64,
    memory: &[u64],
    instr: &JoltONNXBytecode,
    cycle: &JoltONNXCycle,
) {
    let expected_value = memory[address];
    assert_eq!(
        expected_value,
        actual_value,
        "{} READ error at cycle_{}: {:#?} {:#?}; Expected: {}, got: {} at address {}",
        register_name,
        cycle_index,
        instr,
        cycle,
        expected_value as u32 as i32,
        actual_value as u32 as i32,
        address
    );
}
