use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::consts::{LOG_K_CHUNK, XLEN};

const LOG_K: usize = XLEN * 2;

// TODO: Refactor config for jolt-atlas-core use-case

/// Full one-hot parameters with cached derived values.
///
/// This struct is NOT serialized in the proof. It is constructed by the prover
/// and verifier from `OneHotConfig` plus the proof parameters (bytecode_K, ram_K).
#[derive(Clone, Debug, Default)]
pub struct OneHotParams {
    pub log_k_chunk: usize,
    pub k_chunk: usize,
    pub instruction_d: usize,
    instruction_shifts: Vec<usize>,
}

impl OneHotParams {
    /// Construct full OneHotParams from a config and proof parameters.
    ///
    /// This is used by the verifier to reconstruct the full params from
    /// the minimal config stored in the proof.
    pub fn from_config(config: &OneHotConfig) -> Self {
        let log_k_chunk = config.log_k_chunk as usize;

        let instruction_d = LOG_K.div_ceil(log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();

        Self {
            log_k_chunk,
            k_chunk: 1 << log_k_chunk,
            instruction_d,
            instruction_shifts,
        }
    }

    pub fn from_config_and_log_K(config: &OneHotConfig, log_K: usize) -> Self {
        let log_k_chunk = config.log_k_chunk as usize;

        let instruction_d = log_K.div_ceil(log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();

        Self {
            log_k_chunk,
            k_chunk: 1 << log_k_chunk,
            instruction_d,
            instruction_shifts,
        }
    }

    /// Create OneHotParams for the given trace parameters using default config.
    ///
    /// This is a convenience constructor for the prover.
    pub fn new(log_T: usize) -> Self {
        let config = OneHotConfig::new(log_T);
        Self::from_config(&config)
    }

    /// Extract the minimal config for serialization in the proof.
    pub fn to_config(&self) -> OneHotConfig {
        OneHotConfig {
            log_k_chunk: self.log_k_chunk as u8,
        }
    }

    pub fn lookup_index_chunk(&self, index: u64, idx: usize) -> u8 {
        ((index >> self.instruction_shifts[idx]) & (self.k_chunk - 1) as u64) as u8
    }

    pub fn compute_r_address_chunks<F: JoltField>(
        &self,
        r_address: &[F::Challenge],
    ) -> Vec<Vec<F::Challenge>> {
        let r_address = if r_address.len().is_multiple_of(self.log_k_chunk) {
            r_address.to_vec()
        } else {
            [
                &vec![
                    F::Challenge::from(0_u128);
                    self.log_k_chunk - (r_address.len() & (self.log_k_chunk - 1))
                ],
                r_address,
            ]
            .concat()
        };

        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address
            .chunks(self.log_k_chunk)
            .map(|chunk| chunk.to_vec())
            .collect();

        r_address_chunks
    }
}

/// Minimal configuration for one-hot encoding that gets serialized in the proof.
///
/// Contains only the prover's choices. All fields are `u8` to minimize proof size.
/// The verifier validates these choices and reconstructs the full `OneHotParams`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct OneHotConfig {
    /// Log₂ of chunk size for one-hot encoding of address variables.
    ///
    /// This determines how the address space is decomposed into committed RA polynomials.
    /// Each committed RA polynomial handles 2^log_k_chunk addresses. The total number
    /// of committed RA polynomials is LOG_K / log_k_chunk (e.g., 128/8 = 16 for RV64).
    ///
    /// Must be either 4 or 8 currently
    pub log_k_chunk: u8,
}

impl OneHotConfig {
    /// Create a OneHotConfig with default values based on trace length.
    pub fn new(_log_T: usize /*  TODO: rm log_T param */) -> Self {
        let log_k_chunk = 4;
        Self {
            log_k_chunk: log_k_chunk as u8,
        }
    }

    /// Validates that the one-hot configuration is valid.
    ///
    /// This is called by the verifier to ensure the prover hasn't provided
    /// an invalid configuration that would break soundness.
    pub fn validate(&self) -> Result<(), String> {
        // log_k_chunk must be either 4 or 8
        if self.log_k_chunk != 4 && self.log_k_chunk != 8 {
            return Err(format!(
                "log_k_chunk ({}) must be either 4 or 8",
                self.log_k_chunk
            ));
        }

        Ok(())
    }
}

impl Default for OneHotConfig {
    fn default() -> Self {
        Self {
            log_k_chunk: LOG_K_CHUNK as u8,
        }
    }
}
