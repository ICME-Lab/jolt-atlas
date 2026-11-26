use crate::{
    advice::Advice,
    field::{Mixing, F},
    fold::{basis, Column, Fold},
    norm::Claim,
};

use jolt_sumcheck::single::{
    prove_single, verify_single, SingleFormat, SingleInstance, SingleReduction, SingleSumcheckProof,
};
use jolt_sumcheck::ProveRounds;
use jolt_transcript::{Keccak, ProverState, Transcript, VerifierState};
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct Messages {
    fields: Vec<F>,
    rounds: Vec<SingleSumcheckProof<F>>,
}

pub(crate) enum Channel<'a, 'proof> {
    Send(&'a mut ProverState<Keccak>, &'a mut Messages, &'a Advice),
    Check(
        &'a mut VerifierState<'proof, Keccak>,
        &'proof [F],
        &'proof [SingleSumcheckProof<F>],
    ),
}

impl<'a, 'p> Channel<'a, 'p> {
    pub fn checking(transcript: &'a mut VerifierState<'p, Keccak>, messages: &'p Messages) -> Self {
        Self::Check(transcript, &messages.fields, &messages.rounds)
    }

    pub fn advice(&self) -> Option<&'a Advice> {
        match self {
            Self::Send(_, _, advice) => Some(*advice),
            Self::Check(..) => None,
        }
    }

    pub fn finish(self) -> Result<()> {
        if matches!(self, Self::Check(_, fields, rounds) if !fields.is_empty() || !rounds.is_empty())
        {
            return Err("extra proof messages");
        }
        Ok(())
    }

    pub fn sample(&mut self, count: usize, loss: usize) -> Result<Vec<F>> {
        match self {
            Self::Send(t, _, _) => t.mix(count, loss),
            Self::Check(t, _, _) => t.mix(count, loss),
        }
    }

    pub fn values<const N: usize>(&mut self, produce: impl FnOnce() -> [F; N]) -> Result<[F; N]> {
        Ok(self.fields(N, || produce().into())?.try_into().unwrap())
    }

    pub fn fold<'w, const N: usize, const K: usize>(
        &mut self,
        sum: F,
        rounds: usize,
        degree: usize,
        columns: impl FnOnce() -> [Column<'w, F>; N],
        terminal: [usize; K],
        relation: impl Fn(&[F; N]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, [F; K])> {
        let (end, v) = self.fold_many(
            sum,
            rounds,
            degree,
            || Vec::from(columns()),
            &terminal,
            |v| relation(v.try_into().unwrap()),
        )?;
        Ok((end, v.try_into().map_err(|_| "invalid terminal count")?))
    }

    pub fn fold_many<'w>(
        &mut self,
        sum: F,
        rounds: usize,
        degree: usize,
        columns: impl FnOnce() -> Vec<Column<'w, F>>,
        terminal: &[usize],
        relation: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<F>)> {
        let instance = SingleInstance {
            input_claim: sum,
            rounds,
            degree,
            format: SingleFormat::Compressed,
        };
        self.sumcheck(
            &instance,
            terminal.len(),
            || Fold {
                columns: columns(),
                degree,
                relation,
            },
            |fold| {
                let values = fold.terminal();
                terminal.iter().map(|&i| values[i]).collect()
            },
        )
    }

    pub fn sumcheck<W: ProveRounds<F>>(
        &mut self,
        instance: &SingleInstance<F>,
        count: usize,
        witness: impl FnOnce() -> W,
        terminal: impl FnOnce(W) -> Vec<F>,
    ) -> Result<(SingleReduction<F>, Vec<F>)> {
        let (end, values) = match self {
            Self::Send(t, data, _) => {
                let mut witness = witness();
                let (proof, end) = prove_single(&mut witness, instance, *t)
                    .map_err(|_| "could not prove Sumcheck")?;
                data.rounds.push(proof);
                (end, Some(terminal(witness)))
            }
            Self::Check(t, _, proofs) => {
                let (proof, next) = proofs.split_first().ok_or("missing Sumcheck rounds")?;
                *proofs = next;
                (
                    verify_single(instance, proof, *t).map_err(|_| "invalid Sumcheck rounds")?,
                    None,
                )
            }
        };
        Ok((end, self.fields(count, || values.unwrap())?))
    }

    pub fn fields(&mut self, count: usize, produce: impl FnOnce() -> Vec<F>) -> Result<Vec<F>> {
        match self {
            Self::Send(t, data, _) => {
                let values = produce();
                t.append_values(&values);
                data.fields.extend_from_slice(&values);
                Ok(values)
            }
            Self::Check(t, fields, _) => {
                let (result, next) = fields
                    .split_at_checked(count)
                    .ok_or("missing field values")?;
                *fields = next;
                t.append_values(result);
                Ok(result.to_vec())
            }
        }
    }

    pub fn dot(
        &mut self,
        value: F,
        public: Vec<F>,
        witness: impl FnOnce() -> Vec<F>,
    ) -> Result<Claim<F>> {
        let len = public.len();
        let cols = || {
            [
                Column::table(witness(), len, 1),
                Column::table(public.clone(), len, 1),
            ]
        };
        let (end, [v]) = self.fold(value, len.ilog2() as usize, 2, cols, [0], |c| c[0] * c[1])?;
        let f: F = basis(&end.challenges)
            .into_iter()
            .zip(public)
            .map(|(a, b)| a * b)
            .sum();
        if end.output_claim != f * v {
            return Err("bad inner product evaluation");
        }
        Ok(Claim {
            point: end.challenges,
            value: v,
        })
    }
}
