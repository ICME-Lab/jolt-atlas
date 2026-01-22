use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;

use jolt_core::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, opening_proof::OpeningPoint},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
};

use super::{
    Claims, JoltSNARK,
    bytecode::CircuitFlags,
    dag::state_manager::{ProofData, ProofKeys, Proofs},
    pcs::{OpeningId, ReducedOpeningProof, SumcheckId},
    precompiles::PrecompileSNARK,
    witness::{AllCommittedPolynomials, CommittedPolynomial, VirtualPolynomial},
};

impl<F, PCS, FS> CanonicalSerialize for JoltSNARK<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.memory_K.serialize_with_mode(&mut writer, compress)?;
        let guard = AllCommittedPolynomials::initialize();
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.proofs.serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.twist_sumcheck_switch_index
            .serialize_with_mode(&mut writer, compress)?;
        drop(guard);
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.memory_K.serialized_size(compress)
            + self.opening_claims.serialized_size(compress)
            + self.commitments.serialized_size(compress)
            + self.proofs.serialized_size(compress)
            + self.trace_length.serialized_size(compress)
            + self.twist_sumcheck_switch_index.serialized_size(compress)
    }
}

impl<F, PCS, FS> Valid for JoltSNARK<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.opening_claims.check()?;
        self.commitments.check()?;
        self.proofs.check()?;
        self.trace_length.check()?;
        self.memory_K.check()?;
        self.twist_sumcheck_switch_index.check()?;
        Ok(())
    }
}

impl<F, PCS, FS> CanonicalDeserialize for JoltSNARK<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let memory_K = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let guard = AllCommittedPolynomials::initialize();
        let opening_claims = Claims::deserialize_with_mode(&mut reader, compress, validate)?;
        let commitments =
            Vec::<PCS::Commitment>::deserialize_with_mode(&mut reader, compress, validate)?;
        let proofs = Proofs::<F, PCS, FS>::deserialize_with_mode(&mut reader, compress, validate)?;
        let trace_length = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let twist_sumcheck_switch_index =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        drop(guard);

        Ok(Self {
            opening_claims,
            commitments,
            proofs,
            trace_length,
            memory_K,
            twist_sumcheck_switch_index,
        })
    }
}

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }

        Ok(Claims(claims))
    }
}

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::Committed(polynomial, sumcheck_id) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*sumcheck_id as u8).serialize_with_mode(&mut writer, compress)?;
                polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Virtual(polynomial, sumcheck_id) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*sumcheck_id as u8).serialize_with_mode(&mut writer, compress)?;
                polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::Committed(polynomial, _) => polynomial.serialized_size(compress) + 2,
            OpeningId::Virtual(polynomial, _) => polynomial.serialized_size(compress) + 2,
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let opening_type = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let sumcheck_id = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match opening_type {
            0 => {
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            1 => {
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Virtual(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            CommittedPolynomial::TdInc => {
                0u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::InstructionRa(index) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let tag_size = 1u8.serialized_size(compress);
        tag_size
            + match self {
                CommittedPolynomial::InstructionRa(index) => index.serialized_size(compress),
                _ => 0,
            }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let polynomial = match tag {
            0 => CommittedPolynomial::TdInc,
            1 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                CommittedPolynomial::InstructionRa(index)
            }
            _ => return Err(SerializationError::InvalidData),
        };
        Ok(polynomial)
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            VirtualPolynomial::SpartanAz => 0u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::SpartanBz => 1u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::SpartanCz => 2u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::PC => 3u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::NextPC => 4u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::LeftLookupOperand => {
                5u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RightLookupOperand => {
                6u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::Td => 7u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Imm => 8u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts1Value => 9u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts2Value => 10u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts3Value => 11u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::TdWriteValue => 12u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts1Ra => 13u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts2Ra => 14u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts3Ra => 15u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::TdWa => 16u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::LookupOutput => 17u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::InstructionRaf => 18u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::InstructionRafFlag => {
                19u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::InstructionRa => 20u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::RegistersVal => 21u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::OpFlags(flag) => {
                22u8.serialize_with_mode(&mut writer, compress)?;
                (*flag as u8).serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::LookupTableFlag(index) => {
                23u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileA(index) => {
                24u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileB(index) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileC(index) => {
                26u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaAPrecompile(index) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaBPrecompile(index) => {
                28u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaCPrecompile(index) => {
                29u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::ValFinal => 30u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Product => 31u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::WriteLookupOutputToTD => {
                32u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::SelectCond => {
                33u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::SelectRes => {
                34u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::LeftInstructionInput => {
                35u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RightInstructionInput => {
                36u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::TdIncS => {
                37u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::FpLookupRa(index) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::FpLookupRv(index) => {
                39u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::GatherRa(index) => {
                40u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::GatherHammingWeight(index) => {
                41u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let tag_size = 1u8.serialized_size(compress);
        tag_size
            + match self {
                VirtualPolynomial::OpFlags(flag) => (*flag as u8).serialized_size(compress),
                VirtualPolynomial::LookupTableFlag(index)
                | VirtualPolynomial::PrecompileA(index)
                | VirtualPolynomial::PrecompileB(index)
                | VirtualPolynomial::PrecompileC(index)
                | VirtualPolynomial::RaAPrecompile(index)
                | VirtualPolynomial::RaBPrecompile(index)
                | VirtualPolynomial::RaCPrecompile(index)
                | VirtualPolynomial::FpLookupRa(index)
                | VirtualPolynomial::FpLookupRv(index) => index.serialized_size(compress),
                _ => 0,
            }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let polynomial = match tag {
            0 => VirtualPolynomial::SpartanAz,
            1 => VirtualPolynomial::SpartanBz,
            2 => VirtualPolynomial::SpartanCz,
            3 => VirtualPolynomial::PC,
            4 => VirtualPolynomial::NextPC,
            5 => VirtualPolynomial::LeftLookupOperand,
            6 => VirtualPolynomial::RightLookupOperand,
            7 => VirtualPolynomial::Td,
            8 => VirtualPolynomial::Imm,
            9 => VirtualPolynomial::Ts1Value,
            10 => VirtualPolynomial::Ts2Value,
            11 => VirtualPolynomial::Ts3Value,
            12 => VirtualPolynomial::TdWriteValue,
            13 => VirtualPolynomial::Ts1Ra,
            14 => VirtualPolynomial::Ts2Ra,
            15 => VirtualPolynomial::Ts3Ra,
            16 => VirtualPolynomial::TdWa,
            17 => VirtualPolynomial::LookupOutput,
            18 => VirtualPolynomial::InstructionRaf,
            19 => VirtualPolynomial::InstructionRafFlag,
            20 => VirtualPolynomial::InstructionRa,
            21 => VirtualPolynomial::RegistersVal,
            22 => {
                let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::OpFlags(circuit_flag_from_u8(flag)?)
            }
            23 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::LookupTableFlag(index)
            }
            24 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileA(index)
            }
            25 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileB(index)
            }
            26 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileC(index)
            }
            27 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaAPrecompile(index)
            }
            28 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaBPrecompile(index)
            }
            29 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaCPrecompile(index)
            }
            30 => VirtualPolynomial::ValFinal,
            31 => VirtualPolynomial::Product,
            32 => VirtualPolynomial::WriteLookupOutputToTD,
            33 => VirtualPolynomial::SelectCond,
            34 => VirtualPolynomial::SelectRes,
            35 => VirtualPolynomial::LeftInstructionInput,
            36 => VirtualPolynomial::RightInstructionInput,
            37 => VirtualPolynomial::TdIncS,
            38 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::FpLookupRa(index)
            }
            39 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::FpLookupRv(index)
            }
            _ => return Err(SerializationError::InvalidData),
        };
        Ok(polynomial)
    }
}

fn circuit_flag_from_u8(value: u8) -> Result<CircuitFlags, SerializationError> {
    let flag = match value {
        0 => CircuitFlags::LeftOperandIsTs1Value,
        1 => CircuitFlags::RightOperandIsTs2Value,
        2 => CircuitFlags::RightOperandIsImm,
        3 => CircuitFlags::AddOperands,
        4 => CircuitFlags::SubtractOperands,
        5 => CircuitFlags::MultiplyOperands,
        6 => CircuitFlags::WriteLookupOutputToTD,
        7 => CircuitFlags::Assert,
        8 => CircuitFlags::Advice,
        9 => CircuitFlags::Const,
        10 => CircuitFlags::Select,
        11 => CircuitFlags::Halt,
        _ => return Err(SerializationError::InvalidData),
    };
    Ok(flag)
}

impl CanonicalSerialize for ProofKeys {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        (*self as u8).serialized_size(compress)
    }
}

impl Valid for ProofKeys {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for ProofKeys {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        ProofKeys::from_u8(variant).ok_or(SerializationError::InvalidData)
    }
}

impl<F, PCS, FS> CanonicalSerialize for ProofData<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ProofData::SumcheckProof(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
            ProofData::ReducedOpeningProof(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
            ProofData::PrecompileProof(proof) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
            ProofData::FpLookupProof(proof) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(&mut writer, compress)
            }
            ProofData::ZKSumcheckProof(proof) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                // Serialize round_commitments
                (proof.round_commitments.len() as u32).serialize_with_mode(&mut writer, compress)?;
                for commitment in &proof.round_commitments {
                    commitment.serialize_with_mode(&mut writer, compress)?;
                }
                // Serialize compressed_polys
                (proof.compressed_polys.len() as u32).serialize_with_mode(&mut writer, compress)?;
                for poly in &proof.compressed_polys {
                    poly.serialize_with_mode(&mut writer, compress)?;
                }
                Ok(())
            }
            ProofData::ZKBatchOpeningProof(proof) => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                // Serialize combined_commitment
                proof.combined_commitment.serialize_with_mode(&mut writer, compress)?;
                // Serialize evals_at_zero
                (proof.evals_at_zero.len() as u32).serialize_with_mode(&mut writer, compress)?;
                for eval in &proof.evals_at_zero {
                    eval.serialize_with_mode(&mut writer, compress)?;
                }
                // Serialize evals_at_one
                (proof.evals_at_one.len() as u32).serialize_with_mode(&mut writer, compress)?;
                for eval in &proof.evals_at_one {
                    eval.serialize_with_mode(&mut writer, compress)?;
                }
                // Serialize evals_at_challenge
                (proof.evals_at_challenge.len() as u32).serialize_with_mode(&mut writer, compress)?;
                for eval in &proof.evals_at_challenge {
                    eval.serialize_with_mode(&mut writer, compress)?;
                }
                // Serialize batch_proof (optional)
                match &proof.batch_proof {
                    Some(bp) => {
                        true.serialize_with_mode(&mut writer, compress)?;
                        bp.serialize_with_mode(&mut writer, compress)?;
                    }
                    None => {
                        false.serialize_with_mode(&mut writer, compress)?;
                    }
                }
                // Serialize num_vars
                (proof.num_vars as u32).serialize_with_mode(&mut writer, compress)?;
                Ok(())
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            ProofData::SumcheckProof(proof) => proof.serialized_size(compress),
            ProofData::ReducedOpeningProof(proof) => proof.serialized_size(compress),
            ProofData::PrecompileProof(proof) => proof.serialized_size(compress),
            ProofData::FpLookupProof(proof) => proof.serialized_size(compress),
            ProofData::ZKSumcheckProof(proof) => {
                let mut size = 4; // round_commitments length
                for commitment in &proof.round_commitments {
                    size += commitment.serialized_size(compress);
                }
                size += 4; // compressed_polys length
                for poly in &proof.compressed_polys {
                    size += poly.serialized_size(compress);
                }
                size
            }
            ProofData::ZKBatchOpeningProof(proof) => {
                let mut size = proof.combined_commitment.serialized_size(compress);
                size += 4; // evals_at_zero length
                for eval in &proof.evals_at_zero {
                    size += eval.serialized_size(compress);
                }
                size += 4; // evals_at_one length
                for eval in &proof.evals_at_one {
                    size += eval.serialized_size(compress);
                }
                size += 4; // evals_at_challenge length
                for eval in &proof.evals_at_challenge {
                    size += eval.serialized_size(compress);
                }
                size += 1; // bool for Option presence
                if let Some(bp) = &proof.batch_proof {
                    size += bp.serialized_size(compress);
                }
                size += 4; // num_vars
                size
            }
        }
    }
}

impl<F, PCS, FS> Valid for ProofData<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            ProofData::SumcheckProof(proof) => proof.check(),
            ProofData::ReducedOpeningProof(proof) => proof.check(),
            ProofData::PrecompileProof(proof) => proof.check(),
            ProofData::FpLookupProof(proof) => proof.check(),
            ProofData::ZKSumcheckProof(proof) => {
                for commitment in &proof.round_commitments {
                    commitment.check()?;
                }
                for poly in &proof.compressed_polys {
                    poly.check()?;
                }
                Ok(())
            }
            ProofData::ZKBatchOpeningProof(proof) => {
                proof.combined_commitment.check()?;
                for eval in &proof.evals_at_zero {
                    eval.check()?;
                }
                for eval in &proof.evals_at_one {
                    eval.check()?;
                }
                for eval in &proof.evals_at_challenge {
                    eval.check()?;
                }
                if let Some(bp) = &proof.batch_proof {
                    bp.check()?;
                }
                Ok(())
            }
        }
    }
}

impl<F, PCS, FS> CanonicalDeserialize for ProofData<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => {
                let proof =
                    SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ProofData::SumcheckProof(proof))
            }
            1 => {
                let proof = ReducedOpeningProof::<F, PCS, FS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(ProofData::ReducedOpeningProof(proof))
            }
            2 => {
                let proof = PrecompileSNARK::<F, FS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(ProofData::PrecompileProof(proof))
            }
            3 => {
                let proof = super::fp_lookups::FpLookupProof::<F, FS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(ProofData::FpLookupProof(proof))
            }
            4 => {
                // Deserialize ZKSumcheckProof
                let num_commitments = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let mut round_commitments = Vec::with_capacity(num_commitments);
                for _ in 0..num_commitments {
                    round_commitments.push(PCS::Commitment::deserialize_with_mode(&mut reader, compress, validate)?);
                }
                let num_polys = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let mut compressed_polys = Vec::with_capacity(num_polys);
                for _ in 0..num_polys {
                    compressed_polys.push(jolt_core::poly::unipoly::CompressedUniPoly::<F>::deserialize_with_mode(&mut reader, compress, validate)?);
                }
                Ok(ProofData::ZKSumcheckProof(crate::jolt::sumcheck::ZKSumcheckProof {
                    round_commitments,
                    compressed_polys,
                    opening_hints: Vec::new(), // Hints are not serialized
                }))
            }
            5 => {
                // Deserialize ZKBatchOpeningProof
                let combined_commitment = PCS::Commitment::deserialize_with_mode(&mut reader, compress, validate)?;
                let num_evals_at_zero = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let mut evals_at_zero = Vec::with_capacity(num_evals_at_zero);
                for _ in 0..num_evals_at_zero {
                    evals_at_zero.push(F::deserialize_with_mode(&mut reader, compress, validate)?);
                }
                let num_evals_at_one = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let mut evals_at_one = Vec::with_capacity(num_evals_at_one);
                for _ in 0..num_evals_at_one {
                    evals_at_one.push(F::deserialize_with_mode(&mut reader, compress, validate)?);
                }
                let num_evals_at_challenge = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let mut evals_at_challenge = Vec::with_capacity(num_evals_at_challenge);
                for _ in 0..num_evals_at_challenge {
                    evals_at_challenge.push(F::deserialize_with_mode(&mut reader, compress, validate)?);
                }
                // Deserialize optional batch_proof
                let has_batch_proof = bool::deserialize_with_mode(&mut reader, compress, validate)?;
                let batch_proof = if has_batch_proof {
                    Some(PCS::Proof::deserialize_with_mode(&mut reader, compress, validate)?)
                } else {
                    None
                };
                // Deserialize num_vars
                let num_vars = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                Ok(ProofData::ZKBatchOpeningProof(crate::jolt::sumcheck::ZKSumcheckBatchOpeningProof {
                    combined_commitment,
                    evals_at_zero,
                    evals_at_one,
                    evals_at_challenge,
                    batch_proof,
                    num_vars,
                }))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}
