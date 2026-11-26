use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

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
    columns: Vec<Record>,
    cursor: u64,
}

struct Record {
    kind: RangeColumn,
    at: u64,
    fixed: bool,
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
            columns: Vec::new(),
            cursor: 0,
        })
    }

    pub fn column(
        &mut self,
        shape: Shape,
        width: Width,
        source: impl Fn(usize) -> i128,
    ) -> Result<Slot> {
        self.record(shape, width, false, source)
    }

    pub fn planes(
        &mut self,
        shape: Shape,
        width: u8,
        source: impl Fn(usize) -> i128,
    ) -> Result<Slot> {
        self.record(shape, Width::Unsigned(width), true, source)
    }

    fn record(
        &mut self,
        shape: Shape,
        mut width: Width,
        fixed: bool,
        source: impl Fn(usize) -> i128,
    ) -> Result<Slot> {
        let slot = Slot(self.columns.len());
        let start = self.cursor;
        if let Some(file) = &mut self.file {
            if !fixed {
                let mut required = 0;
                for j in 0..shape.rows * shape.cols {
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
                    required = required.max(n as usize);
                }
                required = required.max(usize::from(matches!(width, Width::Signed(_))));
                if required > bits(width) {
                    return Err("advice exceeds declared integer range");
                }
                width = resized(width, required as u8);
            }
            let n = bits(width);
            if n != 0 {
                let len = shape.padded();
                let mut out = vec![0u8; (len * n).div_ceil(8) + 16];
                for i in 0..len {
                    let x = shape.index(i).map_or(0, &source);
                    let encoded =
                        encode(x, width).ok_or("advice exceeds declared integer range")? as u128;
                    let at = i * n / 8;
                    let before = u128::from_le_bytes(out[at..][..16].try_into().unwrap());
                    out[at..][..16]
                        .copy_from_slice(&(before | (encoded << (i * n % 8))).to_le_bytes());
                }
                out.truncate((len * n).div_ceil(8));
                file.write_all(&out).map_err(|_| "could not store advice")?;
                self.cursor += out.len() as u64;
            }
        }
        let kind = RangeColumn {
            vars: shape.bits(),
            width,
        };
        self.columns.push(Record {
            kind,
            at: start,
            fixed,
        });
        Ok(slot)
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
                || (column.fixed && usize::from(width) != bits(allowed))
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

    fn load(&mut self, ids: &[usize]) -> Result<Vec<Vec<u8>>> {
        let file = self.file.as_mut().ok_or("missing prover advice")?;
        ids.iter()
            .map(|&i| {
                let record = &self.columns[i];
                let len = (bits(record.kind.width) * (1 << record.kind.vars)).div_ceil(8);
                let mut buffer = vec![0; len + 16];
                if len != 0 {
                    file.seek(SeekFrom::Start(record.at))
                        .and_then(|_| file.read_exact(&mut buffer[..len]))
                        .map_err(|_| "could not read advice")?;
                }
                Ok(buffer)
            })
            .collect()
    }

    fn commit(&mut self, group: &Group, ids: &[usize]) -> Result<crate::bitz::Committed> {
        let buffers = self.load(ids)?;
        group.commit_detached(|i, cell| self.value(ids[i], cell, &buffers[i]))
    }

    fn value(&self, id: usize, cell: usize, data: &[u8]) -> i128 {
        let width = self.columns[id].kind.width;
        let n = bits(width);
        let bit = cell * n;
        let word = u128::from_le_bytes(data[bit / 8..][..16].try_into().unwrap());
        ((word >> (bit % 8)) & ((1 << n) - 1)) as i128 - offset(width)
    }
}

impl Buckets {
    pub fn commit(
        &self,
        advice: &mut Advice,
        channel: &mut ProverState<Keccak>,
    ) -> Result<Vec<[u8; 32]>> {
        self.0
            .iter()
            .map(|(group, ids)| {
                let root = advice.commit(group, ids)?.root;
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
                let data = advice.load(ids)?;
                let source = |i: usize, cell: usize| advice.value(ids[i], cell, data[i].as_slice());
                let committed = group.commit_detached(source)?;
                if &committed.root != root {
                    return Err("advice commitment changed");
                }
                let job = group.reduce(committed, &queries, channel, Some(source))?;
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
