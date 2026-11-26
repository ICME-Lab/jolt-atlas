use rayon::prelude::*;
use std::{fs::File, io::Write};

use crate::{
    bitz::{Group, Query, RangeColumn},
    channel::Result,
    field::F,
    norm::{Claim, Width},
    tensor::Shape,
};
use jolt_transcript::{Keccak, ProverState, Transcript, VerifierState};

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Slot(usize);

pub(crate) struct Advice {
    file: Option<File>,
    mapped: Option<memmap2::Mmap>,
    columns: Vec<Record>,
    cursor: u64,
}

struct Record {
    kind: RangeColumn,
    packed: Packed,
    digit: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Packed {
    at: usize,
    width: u8,
}

pub(crate) struct Buckets(Vec<(Group, Vec<usize>)>);

pub(crate) struct Requests(pub Vec<(Slot, Claim<F>, Option<Vec<F>>)>);

impl Requests {
    pub fn push(&mut self, slot: Slot, point: &[F], value: F) {
        self.0.push((
            slot,
            Claim {
                point: point.to_vec(),
                value,
            },
            None,
        ));
    }
    pub fn weighted(&mut self, slot: Slot, point: &[F], value: F, planes: Vec<F>) {
        self.0.push((
            slot,
            Claim {
                point: point.to_vec(),
                value,
            },
            Some(planes),
        ));
    }
}

impl Advice {
    pub fn new(proving: bool) -> Result<Self> {
        let file = if proving {
            Some(tempfile::tempfile().map_err(|_| "could not create advice file")?)
        } else {
            None
        };
        Ok(Self {
            file,
            mapped: None,
            columns: Vec::new(),
            cursor: 0,
        })
    }

    pub fn column(
        &mut self,
        shape: Shape,
        width: Width,
        source: impl Fn(usize) -> i128 + Sync,
    ) -> Result<Slot> {
        self.record(shape, width, source)
    }

    pub fn lookup(
        &mut self,
        shape: Shape,
        width: usize,
        source: impl Fn(usize) -> i128 + Sync,
    ) -> Result<Packed> {
        self.pack(shape, Width::Unsigned(width as u8), &source)
    }

    pub fn digit(&mut self, shape: Shape, packed: Packed, shift: usize, width: usize) -> Slot {
        let slot = Slot(self.columns.len());
        self.columns.push(Record {
            kind: RangeColumn {
                vars: shape.bits(),
                width: Width::Unsigned(1 << width),
            },
            packed,
            digit: Some((shift, (1 << width) - 1)),
        });
        slot
    }

    fn record(
        &mut self,
        shape: Shape,
        mut width: Width,
        source: impl Fn(usize) -> i128 + Sync,
    ) -> Result<Slot> {
        let slot = Slot(self.columns.len());
        if self.file.is_some() {
            let required = (0..shape.rows * shape.cols)
                .into_par_iter()
                .with_min_len(8192)
                .map(|j| {
                    let x = source(j);
                    let n = match width {
                        Width::Signed(_) => {
                            129 - if x < 0 {
                                (!x as u128).leading_zeros()
                            } else {
                                (x as u128).leading_zeros()
                            }
                        }
                        Width::Unsigned(_) => 128 - (x as u128).leading_zeros(),
                    };
                    n as usize
                })
                .max()
                .unwrap_or(0);
            let mut required = required;
            required = required.max(usize::from(matches!(width, Width::Signed(_))));
            if required > bits(width) {
                return Err("advice exceeds declared integer range");
            }
            width = resized(width, required as u8);
        }
        let packed = self.pack(shape, width, &source)?;
        self.columns.push(Record {
            kind: RangeColumn {
                vars: shape.bits(),
                width,
            },
            packed,
            digit: None,
        });
        Ok(slot)
    }

    fn pack(
        &mut self,
        shape: Shape,
        width: Width,
        source: &(impl Fn(usize) -> i128 + Sync),
    ) -> Result<Packed> {
        let n = bits(width);
        let packed = Packed {
            at: self.cursor as usize,
            width: n as u8,
        };
        if let Some(file) = &mut self.file {
            if n != 0 {
                let len = shape.padded();
                let mut out = vec![0u8; (len * n).div_ceil(8)];
                out.par_chunks_mut(128 * n).enumerate().try_for_each(
                    |(block, chunk)| -> Result<()> {
                        let start = block << 10;
                        let mut cursor = 0;
                        let mut word = 0u128;
                        let mut used = 0;
                        for i in start..(start + 1024).min(len) {
                            let x = shape.index(i).map_or(0, source);
                            let value = encode(x, width)
                                .ok_or("advice exceeds declared integer range")?
                                as u128;
                            word |= value << used;
                            used += n;
                            while used >= 64 {
                                chunk[cursor..cursor + 8]
                                    .copy_from_slice(&(word as u64).to_le_bytes());
                                cursor += 8;
                                used -= 64;
                                word = if used == 0 { 0 } else { value >> (n - used) };
                            }
                        }
                        chunk[cursor..].copy_from_slice(&word.to_le_bytes()[..used.div_ceil(8)]);
                        Ok(())
                    },
                )?;
                file.write_all(&out).map_err(|_| "could not store advice")?;
                self.cursor += out.len() as u64;
            }
        }
        Ok(packed)
    }

    pub fn widths(&self) -> Vec<u8> {
        self.columns
            .iter()
            .map(|x| bits(x.kind.width) as u8)
            .collect()
    }

    pub fn bind_widths(&mut self, widths: &[u8]) -> Result<()> {
        if self.columns.len() != widths.len() {
            return Err("invalid number of witness widths");
        }
        for (column, &width) in self.columns.iter_mut().zip(widths) {
            let allowed = column.kind.width;
            if usize::from(width) > bits(allowed)
                || (column.digit.is_some() && usize::from(width) != bits(allowed))
                || (matches!(allowed, Width::Signed(_)) && width == 0)
            {
                return Err("invalid witness width");
            }
            column.kind.width = resized(allowed, width);
        }
        Ok(())
    }

    pub fn buckets(&self) -> Result<Buckets> {
        const MAX: usize = 32 << 23;
        let mut sizes = Vec::<(usize, Vec<usize>)>::new();
        let mut amounts: Vec<_> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let n = (1 << c.kind.vars.max(9)) * bits(c.kind.width);
                (i, n)
            })
            .collect();
        amounts.sort_unstable_by_key(|&(i, n)| (std::cmp::Reverse(n), i));
        for (i, n) in amounts {
            if let Some((used, ids)) = sizes.iter_mut().find(|(used, _)| *used + n <= MAX) {
                *used += n;
                ids.push(i);
            } else {
                sizes.push((n, vec![i]));
            }
        }
        let groups = sizes
            .into_iter()
            .map(|(_, ids)| {
                let columns: Vec<_> = ids.iter().map(|&i| self.columns[i].kind).collect();
                Ok((Group::new(&columns)?, ids))
            })
            .collect::<Result<_>>()?;
        Ok(Buckets(groups))
    }

    fn seal(&mut self) -> Result<()> {
        let mut file = self.file.take().ok_or("missing prover advice")?;
        file.write_all(&[0; 16])
            .map_err(|_| "could not finish advice file")?;
        // The temporary file is private, and its only writable handle is dropped here.
        let mapped =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|_| "could not map prover advice")?;
        self.mapped = Some(mapped);
        Ok(())
    }

    pub fn unpack(&self, packed: Packed, cell: usize) -> u128 {
        let n = usize::from(packed.width);
        if n == 0 {
            return 0;
        }
        let data = self.mapped.as_ref().unwrap();
        let bit = cell * n;
        let at = packed.at + bit / 8;
        let word = u128::from_le_bytes(data[at..][..16].try_into().unwrap());
        (word >> (bit % 8)) & ((1 << n) - 1)
    }

    pub fn value(&self, slot: Slot, cell: usize) -> i128 {
        let record = &self.columns[slot.0];
        let v = self.unpack(record.packed, cell);
        match record.digit {
            Some((shift, mask)) => 1i128 << ((v as usize >> shift) & mask),
            None => v as i128 - offset(record.kind.width),
        }
    }

    fn source<'a>(&'a self, ids: &'a [usize]) -> impl Fn(usize, usize) -> i128 + Sync + 'a {
        move |slot, cell| self.value(Slot(ids[slot]), cell)
    }
}

impl Buckets {
    pub fn commit(
        &self,
        advice: &mut Advice,
        channel: &mut ProverState<Keccak>,
    ) -> Result<Vec<[u8; 32]>> {
        advice.seal()?;
        self.0
            .iter()
            .map(|(group, ids)| {
                let root = group.commit_detached(advice.source(ids))?.root;
                channel.append_bytes(&root);
                Ok(root)
            })
            .collect()
    }

    pub fn bind(&self, roots: &[[u8; 32]], channel: &mut VerifierState<'_, Keccak>) -> Result<()> {
        if self.0.len() != roots.len() {
            return Err("invalid number of BitZ groups");
        }
        for ((group, _), root) in self.0.iter().zip(roots) {
            group.bind_root(*root, channel);
        }
        Ok(())
    }

    pub fn prove(
        &self,
        advice: &mut Advice,
        roots: &[[u8; 32]],
        requests: Requests,
        channel: &mut ProverState<Keccak>,
        mut emit: impl FnMut(crate::bitz::Proof) -> Result<()>,
    ) -> Result<()> {
        let requests = self.distribute(advice.columns.len(), requests)?;
        std::thread::scope(|threads| -> Result<()> {
            let queue = if self.0.len() > 1
                && rayon::current_thread_index().is_none()
                && rayon::current_num_threads() > 1
            {
                let (send, jobs) = std::sync::mpsc::sync_channel::<crate::bitz::Pending<'_>>(0);
                let (done, receive) = std::sync::mpsc::sync_channel(1);
                std::thread::Builder::new()
                    .spawn_scoped(threads, move || {
                        let mut cache = std::collections::HashMap::new();
                        for job in jobs {
                            if done.send(job.open(&mut cache)).is_err() {
                                break;
                            }
                        }
                    })
                    .ok()
                    .map(|_| (send, receive))
            } else {
                None
            };
            let mut cache = std::collections::HashMap::new();
            for (i, (((group, ids), root), queries)) in
                self.0.iter().zip(roots).zip(requests).enumerate()
            {
                let source = advice.source(ids);
                let committed = group.commit_detached(&source)?;
                if &committed.root != root {
                    return Err("advice commitment changed");
                }
                let job = group.reduce(committed, &queries, channel, source)?;
                if let Some((send, receive)) = &queue {
                    if i != 0 {
                        emit(receive.recv().map_err(|_| "BitZ worker stopped")??)?;
                    }
                    send.send(job).map_err(|_| "BitZ worker stopped")?;
                } else {
                    emit(job.open(&mut cache)?)?;
                }
            }
            if let Some((send, receive)) = queue {
                drop(send);
                emit(receive.recv().map_err(|_| "BitZ worker stopped")??)?;
            }
            Ok(())
        })
    }

    pub fn verify<'p>(
        &self,
        advice: &Advice,
        requests: Requests,
        roots: &[[u8; 32]],
        channel: &mut VerifierState<'_, Keccak>,
        mut evaluation: impl FnMut(usize) -> Result<std::borrow::Cow<'p, crate::bitz::Proof>>,
    ) -> Result<()> {
        let requests = self.distribute(advice.columns.len(), requests)?;
        for (i, (((group, _), &root), queries)) in
            self.0.iter().zip(roots).zip(requests).enumerate()
        {
            let item = evaluation(i)?;
            group.verify(root, &queries, &item, channel)?;
        }
        Ok(())
    }

    fn distribute(&self, count: usize, requests: Requests) -> Result<Vec<Vec<Query>>> {
        let mut position = vec![(0, 0); count];
        for (part, (_, ids)) in self.0.iter().enumerate() {
            for (slot, &index) in ids.iter().enumerate() {
                position[index] = (part, slot);
            }
        }
        let mut parts: Vec<Vec<Query>> = (0..self.0.len()).map(|_| Vec::new()).collect();
        for (Slot(id), claim, planes) in requests.0 {
            let &(part, slot) = position.get(id).ok_or("invalid advice slot")?;
            parts[part].push(Query {
                slot,
                claim,
                planes,
            });
        }
        for (part, (_, ids)) in parts.iter_mut().zip(&self.0) {
            if part.is_empty() && !ids.is_empty() {
                return Err("unconstrained BitZ group");
            }
        }
        Ok(parts)
    }
}

fn bits(width: Width) -> usize {
    match width {
        Width::Signed(n) | Width::Unsigned(n) => n.into(),
    }
}
fn resized(width: Width, bits: u8) -> Width {
    match width {
        Width::Signed(_) => Width::Signed(bits),
        Width::Unsigned(_) => Width::Unsigned(bits),
    }
}
fn offset(width: Width) -> i128 {
    match width {
        Width::Signed(n) => 1 << (n - 1),
        Width::Unsigned(_) => 0,
    }
}
fn encode(x: i128, width: Width) -> Option<i128> {
    let n = x.checked_add(offset(width))?;
    (0..1i128 << bits(width)).contains(&n).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_transcript::{protocol_tag, prover_transcript};

    #[test]
    fn packed_lookup_commits_to_the_same_one_hot_columns() {
        let mut packed = Advice::new(true).unwrap();
        let mut explicit = Advice::new(true).unwrap();
        let shape = Shape::new(8, 8).unwrap();
        let key = packed.lookup(shape, 6, |i| i as i128).unwrap();
        for shift in [0, 3] {
            packed.digit(shape, key, shift, 3);
            explicit
                .column(shape, Width::Unsigned(8), |i| 1 << ((i >> shift) & 7))
                .unwrap();
        }
        let roots = |advice: &mut Advice| {
            let tag = protocol_tag(b"lookup-columns");
            let mut transcript = prover_transcript::<Keccak>(&tag, b"");
            advice
                .buckets()
                .unwrap()
                .commit(advice, &mut transcript)
                .unwrap()
        };
        assert_eq!(packed.widths(), explicit.widths());
        assert_eq!(roots(&mut packed), roots(&mut explicit));
    }
}
