use crate::jolt::{
    bytecode::{CircuitFlags, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS},
    dag::state_manager::StateManager,
    executor::instructions::InstructionLookup,
    lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
    pcs::SumcheckId,
    trace::WORD_SIZE,
    witness::VirtualPolynomial,
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;
use std::iter::once;
use strum::{EnumCount, IntoEnumIterator};

pub struct ReadRafCheck<F: JoltField> {
    _marker: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone, Copy)]
enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Registers read-write sumcheck
    Stage2,
    /// Registers val sumcheck wa, PCSumcheck, Instruction Lookups
    Stage3,
    /// Spartan product
    Stage4,
    /// Spartan Instruction
    Stage5,
}

impl<F: JoltField> ReadRafCheck<F> {
    /// Updates transcript to match verifier
    #[tracing::instrument(skip_all, name = "BytecodeReadRafCheck::prove")]
    pub fn prove(sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>) {
        #[cfg(test)]
        {
            // rm openings for appended_virtual_openings for testing purposes
            let _ =
                sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
            let _ =
                sm.get_virtual_polynomial_opening(VirtualPolynomial::Td, SumcheckId::SpartanOuter);
            CircuitFlags::iter().for_each(|flag| {
                let _ = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::OpFlags(flag),
                    SumcheckId::SpartanOuter,
                );
            });
            std::iter::empty()
                .chain(once(VirtualPolynomial::TdWa))
                .chain(once(VirtualPolynomial::Ts1Ra))
                .chain(once(VirtualPolynomial::Ts2Ra))
                .chain(once(VirtualPolynomial::Ts3Ra))
                .for_each(|vp| {
                    let _ = sm
                        .get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking);
                });
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::TdWa,
                SumcheckId::RegistersValEvaluation,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            );
            (0..LookupTables::<WORD_SIZE>::COUNT).for_each(|i| {
                let _ = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(i),
                    SumcheckId::InstructionReadRaf,
                );
            });
            let _ =
                sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
            let _ =
                sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);

            // stage 2
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::Td,
                SumcheckId::WriteLookupOutputToTDVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToTD),
                SumcheckId::WriteLookupOutputToTDVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Select),
                SumcheckId::SelectCondVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Select),
                SumcheckId::SelectResVirtualization,
            );

            // stage 3
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsTs1Value),
                SumcheckId::InstructionInputVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsTs2Value),
                SumcheckId::InstructionInputVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
                SumcheckId::InstructionInputVirtualization,
            );
            let _ = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::Imm,
                SumcheckId::InstructionInputVirtualization,
            );
        }
        let _gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let _gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let _gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let _gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let _gamma: F = sm.transcript.borrow_mut().challenge_scalar();
    }

    pub fn verify(sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>) {
        let K = sm.get_verifier_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let (val_1, rv_claim_1, r_cycle_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let (val_2, rv_claim_2, r_cycle_2) = Self::compute_val_rv(sm, ReadCheckingValType::Stage2);
        let (val_3, rv_claim_3, r_cycle_3) = Self::compute_val_rv(sm, ReadCheckingValType::Stage3);
        let (val_4, rv_claim_4, r_cycle_4) = Self::compute_val_rv(sm, ReadCheckingValType::Stage4);
        let (val_5, rv_claim_5, r_cycle_5) = Self::compute_val_rv(sm, ReadCheckingValType::Stage5);
        let int_poly = IdentityPolynomial::new(log_K);
        assert_eq!(r_cycle_1.len(), r_cycle_2.len());
        assert_eq!(r_cycle_1.len(), r_cycle_3.len());
        let val_1 = MultilinearPolynomial::from(val_1);
        let val_2 = MultilinearPolynomial::from(val_2);
        let val_3 = MultilinearPolynomial::from(val_3);
        let val_4 = MultilinearPolynomial::from(val_4);
        let val_5 = MultilinearPolynomial::from(val_5);

        // Check the rv bytecode claims
        assert_eq!(rv_claim_1, val_1.evaluate(&r_cycle_1));
        assert_eq!(rv_claim_2, val_2.evaluate(&r_cycle_2));
        assert_eq!(rv_claim_3, val_3.evaluate(&r_cycle_3));
        assert_eq!(rv_claim_4, val_4.evaluate(&r_cycle_4));
        assert_eq!(rv_claim_5, val_5.evaluate(&r_cycle_5));

        // Check tdwa
        {
            let (r, claim) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::TdWa,
                SumcheckId::RegistersValEvaluation,
            );
            let r_register = sm
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::TdWa,
                    SumcheckId::RegistersValEvaluation,
                )
                .0
                .r;
            let r_register: Vec<_> = r_register[..(sm.get_memory_K()).log_2()].to_vec();
            let eq_r_register = EqPolynomial::evals(&r_register);
            debug_assert_eq!(eq_r_register.len(), sm.get_memory_K());
            let poly = MultilinearPolynomial::from(
                sm.get_bytecode()
                    .par_iter()
                    .map(|instruction| eq_r_register[instruction.td as usize])
                    .collect::<Vec<F>>(),
            );
            let (_, r_cycle) = r.split_at((sm.get_memory_K()).log_2());
            assert_eq!(claim, poly.evaluate(&r_cycle.r));
        }

        // Check the raf openings
        let (raf_point_1, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        assert_eq!(raf_claim, int_poly.evaluate(&raf_point_1.r));
        let (raf_point_2, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        assert_eq!(raf_shift_claim, int_poly.evaluate(&raf_point_2.r));
    }

    fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> (Vec<F>, F, Vec<F>) {
        match val_type {
            ReadCheckingValType::Stage1 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_CIRCUIT_FLAGS + 1 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Imm,
                    SumcheckId::SpartanOuter,
                );
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
            ReadCheckingValType::Stage2 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..3 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Ts1Ra,
                    SumcheckId::RegistersReadWriteChecking,
                );
                let (_, r_cycle) = r.split_at((sm.get_memory_K()).log_2());
                (
                    Self::compute_val_2(sm, &gamma_powers),
                    Self::compute_rv_claim_2(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
            ReadCheckingValType::Stage3 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_LOOKUP_TABLES + 1 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::InstructionRafFlag,
                    SumcheckId::InstructionReadRaf,
                );
                (
                    Self::compute_val_3(sm, &gamma_powers),
                    Self::compute_rv_claim_3(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
            ReadCheckingValType::Stage4 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..2 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Td,
                    SumcheckId::WriteLookupOutputToTDVirtualization,
                );
                (
                    Self::compute_val_4(sm, &gamma_powers),
                    Self::compute_rv_claim_4(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
            ReadCheckingValType::Stage5 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..3 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Imm,
                    SumcheckId::InstructionInputVirtualization,
                );
                (
                    Self::compute_val_5(sm, &gamma_powers),
                    Self::compute_rv_claim_5(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
        }
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = imm(k)
    ///             + gamma * circuit_flags[0](k) + gamma^2 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(instruction.imm);
                linear_combination += instruction.td.field_mul(gamma_powers[1]);
                for (flag, gamma_power) in instruction
                    .circuit_flags()
                    .iter()
                    .zip(gamma_powers[2..].iter())
                {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, imm_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Td, SumcheckId::SpartanOuter);
        once(imm_claim)
            .chain(once(rd_claim))
            .chain(CircuitFlags::iter().map(|flag| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::OpFlags(flag),
                    SumcheckId::SpartanOuter,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register) + gamma * rs1(k, r_register) + gamma^2 * rs2(k, r_register)
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// and analogously for rs1(k, k') and rs2(k, k').
    /// This particular Val virtualizes claims output by the registers read/write checking sumcheck.
    fn compute_val_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        let K = sm.get_memory_K();
        let r_register = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::TdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        let r_register = &r_register[..(K).log_2()];
        let eq_r_register = EqPolynomial::evals(r_register);
        debug_assert_eq!(eq_r_register.len(), K);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                std::iter::empty()
                    .chain(once(instruction.td))
                    .chain(once(instruction.ts1))
                    .chain(once(instruction.ts2))
                    .chain(once(instruction.ts3))
                    .map(|r| eq_r_register[r as usize])
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum::<F>()
            })
            .collect()
    }

    fn compute_rv_claim_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        std::iter::empty()
            .chain(once(VirtualPolynomial::TdWa))
            .chain(once(VirtualPolynomial::Ts1Ra))
            .chain(once(VirtualPolynomial::Ts2Ra))
            .chain(once(VirtualPolynomial::Ts3Ra))
            .map(|vp| {
                sm.get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register) + gamma * instr_raf_flag(k)
    ///             + gamma^2 * lookup_table_flag[0](k)
    ///             + gamma^3 * lookup_table_flag[1](k) + ...
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes claims output by the PCSumcheck,
    /// the registers val-evaluation sumcheck, and the instruction lookups sumcheck.
    fn compute_val_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let flags = instruction.circuit_flags();

                let mut linear_combination: F = F::zero();
                if !flags.is_interleaved_operands() {
                    linear_combination += gamma_powers[0];
                }

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[1 + table_index];
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, raf_flag_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        std::iter::empty()
            .chain(once(raf_flag_claim))
            .chain((0..LookupTables::<WORD_SIZE>::COUNT).map(|i| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(i),
                    SumcheckId::InstructionReadRaf,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    fn compute_val_4(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let flags = instruction.circuit_flags();
                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(instruction.td);
                if flags[CircuitFlags::WriteLookupOutputToTD as usize] {
                    linear_combination += gamma_powers[1];
                }
                if flags[CircuitFlags::Select as usize] {
                    linear_combination += gamma_powers[2];
                }
                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_4(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, td_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::Td,
            SumcheckId::WriteLookupOutputToTDVirtualization,
        );
        let (_, td_flag_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToTD),
            SumcheckId::WriteLookupOutputToTDVirtualization,
        );
        let (_, select_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Select),
            SumcheckId::SelectCondVirtualization,
        );

        once(td_claim)
            .chain(once(td_flag_claim))
            .chain(once(select_claim))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    fn compute_val_5(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let flags = instruction.circuit_flags();
                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(instruction.imm);
                if flags[CircuitFlags::LeftOperandIsTs1Value as usize] {
                    linear_combination += gamma_powers[1];
                }
                if flags[CircuitFlags::RightOperandIsTs2Value as usize] {
                    linear_combination += gamma_powers[2];
                }
                if flags[CircuitFlags::RightOperandIsImm as usize] {
                    linear_combination += gamma_powers[3];
                }
                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_5(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, imm) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_flag) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsTs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_flag) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsTs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_flag_is_imm) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );
        once(imm)
            .chain(once(left_flag))
            .chain(once(right_flag))
            .chain(once(right_flag_is_imm))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }
}
