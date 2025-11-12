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
        self._bytecode_d
            .serialize_with_mode(&mut writer, compress)?;
        let guard = AllCommittedPolynomials::initialize(self._bytecode_d);
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
            + self._bytecode_d.serialized_size(compress)
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
        self._bytecode_d.check()?;
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
        let bytecode_d = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let guard = AllCommittedPolynomials::initialize(bytecode_d);
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
            _bytecode_d: bytecode_d,
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
            CommittedPolynomial::LeftInstructionInput => {
                0u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::RightInstructionInput => {
                1u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::Product => {
                2u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::SelectCond => {
                3u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::SelectRes => {
                4u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::WriteLookupOutputToTD => {
                5u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::TdInc => {
                6u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::TdIncS => {
                7u8.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::InstructionRa(index) => {
                8u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            CommittedPolynomial::BytecodeRa(index) => {
                9u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let tag_size = 1u8.serialized_size(compress);
        tag_size
            + match self {
                CommittedPolynomial::InstructionRa(index)
                | CommittedPolynomial::BytecodeRa(index) => index.serialized_size(compress),
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
            0 => CommittedPolynomial::LeftInstructionInput,
            1 => CommittedPolynomial::RightInstructionInput,
            2 => CommittedPolynomial::Product,
            3 => CommittedPolynomial::SelectCond,
            4 => CommittedPolynomial::SelectRes,
            5 => CommittedPolynomial::WriteLookupOutputToTD,
            6 => CommittedPolynomial::TdInc,
            7 => CommittedPolynomial::TdIncS,
            8 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                CommittedPolynomial::InstructionRa(index)
            }
            9 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                CommittedPolynomial::BytecodeRa(index)
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
            VirtualPolynomial::UnexpandedPC => 4u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::NextPC => 5u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::NextUnexpandedPC => {
                6u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::NextIsNoop => {
                7u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::LeftLookupOperand => {
                8u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RightLookupOperand => {
                9u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::Td => 10u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Imm => 11u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts1Value => 12u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts2Value => 13u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts3Value => 14u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::TdWriteValue => 15u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts1Ra => 16u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts2Ra => 17u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::Ts3Ra => 18u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::TdWa => 19u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::LookupOutput => 20u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::InstructionRaf => 21u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::InstructionRafFlag => {
                22u8.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::InstructionRa => 23u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::RegistersVal => 24u8.serialize_with_mode(&mut writer, compress)?,
            VirtualPolynomial::OpFlags(flag) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                (*flag as u8).serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::LookupTableFlag(index) => {
                26u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileA(index) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileB(index) => {
                28u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::PrecompileC(index) => {
                29u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaAPrecompile(index) => {
                30u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaBPrecompile(index) => {
                31u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::RaCPrecompile(index) => {
                32u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(&mut writer, compress)?;
            }
            VirtualPolynomial::ValFinal => 33u8.serialize_with_mode(&mut writer, compress)?,
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
                | VirtualPolynomial::RaCPrecompile(index) => index.serialized_size(compress),
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
            4 => VirtualPolynomial::UnexpandedPC,
            5 => VirtualPolynomial::NextPC,
            6 => VirtualPolynomial::NextUnexpandedPC,
            7 => VirtualPolynomial::NextIsNoop,
            8 => VirtualPolynomial::LeftLookupOperand,
            9 => VirtualPolynomial::RightLookupOperand,
            10 => VirtualPolynomial::Td,
            11 => VirtualPolynomial::Imm,
            12 => VirtualPolynomial::Ts1Value,
            13 => VirtualPolynomial::Ts2Value,
            14 => VirtualPolynomial::Ts3Value,
            15 => VirtualPolynomial::TdWriteValue,
            16 => VirtualPolynomial::Ts1Ra,
            17 => VirtualPolynomial::Ts2Ra,
            18 => VirtualPolynomial::Ts3Ra,
            19 => VirtualPolynomial::TdWa,
            20 => VirtualPolynomial::LookupOutput,
            21 => VirtualPolynomial::InstructionRaf,
            22 => VirtualPolynomial::InstructionRafFlag,
            23 => VirtualPolynomial::InstructionRa,
            24 => VirtualPolynomial::RegistersVal,
            25 => {
                let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::OpFlags(circuit_flag_from_u8(flag)?)
            }
            26 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::LookupTableFlag(index)
            }
            27 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileA(index)
            }
            28 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileB(index)
            }
            29 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::PrecompileC(index)
            }
            30 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaAPrecompile(index)
            }
            31 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaBPrecompile(index)
            }
            32 => {
                let index = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                VirtualPolynomial::RaCPrecompile(index)
            }
            33 => VirtualPolynomial::ValFinal,
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
        7 => CircuitFlags::InlineSequenceInstruction,
        8 => CircuitFlags::Assert,
        9 => CircuitFlags::DoNotUpdateUnexpandedPC,
        10 => CircuitFlags::Advice,
        11 => CircuitFlags::Const,
        12 => CircuitFlags::IsNoop,
        13 => CircuitFlags::Select,
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
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            ProofData::SumcheckProof(proof) => proof.serialized_size(compress),
            ProofData::ReducedOpeningProof(proof) => proof.serialized_size(compress),
            ProofData::PrecompileProof(proof) => proof.serialized_size(compress),
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
            _ => Err(SerializationError::InvalidData),
        }
    }
}
