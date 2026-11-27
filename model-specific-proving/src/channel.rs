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
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, Keccak, ProverState, Transcript,
    VerifierState,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct Messages {
    fields: Vec<F>,
    rounds: Vec<SingleSumcheckProof<F>>,
    branches: Vec<Branch>,
    windows: Vec<Window>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Branch {
    fields: Vec<F>,
    rounds: Vec<SingleSumcheckProof<F>>,
    bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Window {
    fields: Vec<F>,
    rounds: Vec<SingleSumcheckProof<F>>,
    branches: Vec<Branch>,
    bytes: Vec<u8>,
    pub bitz: Vec<u8>,
}

impl Window {
    pub fn prove(
        seed: [u8; 32],
        lookups: crate::lookup::Lookups<'_>,
        advice: &Advice,
    ) -> Result<(Self, [u8; 32], crate::advice::Requests)> {
        let tag = protocol_tag(b"atlas/lookup-window/v1");
        let mut t = prover_transcript::<Keccak>(&tag, &seed);
        let mut state = Messages::default();
        let mut requests = crate::advice::Requests(Vec::new());
        lookups.finish(
            &mut Channel::Send(&mut t, &mut state, advice),
            &mut requests,
        )?;
        let digest = t.verifier_message_as(32, |v| v.try_into().unwrap());
        Ok((
            Self {
                fields: state.fields,
                rounds: state.rounds,
                branches: state.branches,
                bytes: t.narg_string().to_vec(),
                bitz: Vec::new(),
            },
            digest,
            requests,
        ))
    }

    pub fn verify(
        &self,
        seed: [u8; 32],
        lookups: crate::lookup::Lookups<'_>,
    ) -> Result<([u8; 32], crate::advice::Requests)> {
        let tag = protocol_tag(b"atlas/lookup-window/v1");
        let mut t = verifier_transcript::<Keccak>(&tag, &seed, &self.bytes);
        let mut io = Channel::Check(&mut t, &self.fields, &self.rounds, &self.branches, &[]);
        let mut requests = crate::advice::Requests(Vec::new());
        lookups.finish(&mut io, &mut requests)?;
        io.finish()?;
        let digest = t.verifier_message_as(32, |v| v.try_into().unwrap());
        t.check_eof().map_err(|_| "extra window transcript bytes")?;
        Ok((digest, requests))
    }
}

pub(crate) enum Channel<'a, 'proof> {
    Send(&'a mut ProverState<Keccak>, &'a mut Messages, &'a Advice),
    Check(
        &'a mut VerifierState<'proof, Keccak>,
        &'proof [F],
        &'proof [SingleSumcheckProof<F>],
        &'proof [Branch],
        &'proof [Window],
    ),
}

impl<'a, 'p> Channel<'a, 'p> {
    pub fn checking(transcript: &'a mut VerifierState<'p, Keccak>, messages: &'p Messages) -> Self {
        Self::Check(
            transcript,
            &messages.fields,
            &messages.rounds,
            &messages.branches,
            &messages.windows,
        )
    }

    pub fn parallel<T: Send>(
        &mut self,
        domain: &[u8],
        count: usize,
        work: impl Fn(usize, &mut Channel<'_, '_>) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        if count == 1 {
            return Ok(vec![work(0, self)?]);
        }
        let tag = protocol_tag(domain);
        let block = |bytes: &[u8]| <[u8; 32]>::try_from(bytes).unwrap();
        match self {
            Self::Send(parent, data, advice) => {
                let seed = parent.verifier_message_as(32, block);
                let advice = *advice;
                let children = (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let mut child = prover_transcript::<Keccak>(&tag, &seed);
                        child.append(&[count as u64, i as u64]);
                        let mut messages = Messages::default();
                        let value = work(i, &mut Channel::Send(&mut child, &mut messages, advice))?;
                        let digest = child.verifier_message_as(32, block);
                        let branch = Branch {
                            fields: messages.fields,
                            rounds: messages.rounds,
                            bytes: child.narg_string().to_vec(),
                        };
                        if !messages.branches.is_empty() {
                            return Err("nested parallel proof");
                        }
                        Ok((value, branch, digest))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(children
                    .into_iter()
                    .map(|(value, branch, digest)| {
                        parent.append(&digest);
                        data.branches.push(branch);
                        value
                    })
                    .collect())
            }
            Self::Check(parent, _, _, branches, _) => {
                let (children, rest) = branches
                    .split_at_checked(count)
                    .ok_or("missing proof branches")?;
                *branches = rest;
                let seed = parent.verifier_message_as(32, block);
                children
                    .iter()
                    .enumerate()
                    .map(|(i, branch)| {
                        let mut child = verifier_transcript::<Keccak>(&tag, &seed, &branch.bytes);
                        child.append(&[count as u64, i as u64]);
                        let mut io =
                            Channel::Check(&mut child, &branch.fields, &branch.rounds, &[], &[]);
                        let value = work(i, &mut io)?;
                        io.finish()?;
                        parent.append(&child.verifier_message_as(32, block));
                        child
                            .check_eof()
                            .map_err(|_| "extra branch transcript bytes")?;
                        Ok(value)
                    })
                    .collect()
            }
        }
    }

    pub fn window_seed(&mut self, first: usize, last: usize) -> [u8; 32] {
        let seed =
            |t: &mut ProverState<Keccak>| t.verifier_message_as(32, |v| v.try_into().unwrap());
        match self {
            Self::Send(t, ..) => {
                t.append(&[first as u64, last as u64]);
                seed(*t)
            }
            Self::Check(t, ..) => {
                t.append(&[first as u64, last as u64]);
                t.verifier_message_as(32, |v| v.try_into().unwrap())
            }
        }
    }

    pub fn take_window(&mut self) -> Result<&'p Window> {
        let Self::Check(_, _, _, _, windows) = self else {
            return Err("invalid window reader");
        };
        let (window, next) = windows.split_first().ok_or("missing Lookup window")?;
        *windows = next;
        Ok(window)
    }

    pub fn push_window(&mut self, proof: Window) {
        if let Self::Send(_, state, _) = self {
            state.windows.push(proof);
        }
    }

    pub fn bind_windows(&mut self, roots: &[[u8; 32]]) {
        for root in roots {
            match self {
                Self::Send(t, ..) => t.append(root),
                Self::Check(t, ..) => t.append(root),
            }
        }
    }

    pub fn advice(&self) -> Option<&'a Advice> {
        match self {
            Self::Send(_, _, advice) => Some(*advice),
            Self::Check(..) => None,
        }
    }

    pub fn finish(self) -> Result<()> {
        if matches!(self, Self::Check(_, fields, rounds, branches, windows) if !windows.is_empty() || !fields.is_empty() || !rounds.is_empty() || !branches.is_empty())
        {
            return Err("extra proof messages");
        }
        Ok(())
    }

    pub fn sample(&mut self, count: usize, loss: usize) -> Result<Vec<F>> {
        match self {
            Self::Send(t, _, _) => t.mix(count, loss),
            Self::Check(t, ..) => t.mix(count, loss),
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
            Self::Check(t, _, proofs, _, _) => {
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
            Self::Check(t, fields, ..) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_reordered_parallel_branches() {
        fn work(index: usize, io: &mut Channel<'_, '_>) -> Result<F> {
            let expected = io.sample(2, 2)?[index];
            let [actual] = io.values(|| [expected])?;
            if actual != expected {
                return Err("incorrect branch evaluation");
            }
            Ok(actual)
        }

        let tag = protocol_tag(b"atlas/parallel-test");
        let domain = b"atlas/test-branches";
        let advice = Advice::new(false).unwrap();
        let mut messages = Messages::default();
        let mut prover = prover_transcript::<Keccak>(&tag, b"");
        Channel::Send(&mut prover, &mut messages, &advice)
            .parallel(domain, 2, work)
            .unwrap();

        let verify = |messages: &Messages| -> Result<()> {
            let mut transcript = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
            let mut io = Channel::checking(&mut transcript, messages);
            io.parallel(domain, 2, work)?;
            io.finish()?;
            transcript.check_eof().map_err(|_| "extra transcript bytes")
        };
        assert!(verify(&messages).is_ok());
        messages.branches.swap(0, 1);
        assert!(verify(&messages).is_err());
    }
}
