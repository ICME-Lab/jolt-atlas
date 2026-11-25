use super::*;

const GROUP_BITS: usize = 32 << 23;
const MIN_VARS: usize = world_common::shape::MIN_LOG_BITS;

pub struct Plan {
    pub(super) slots: Vec<(usize, usize)>,
    pub(super) groups: Vec<(BitZParams<Fq>, Pcs)>,
}

fn span(slot: Slot) -> Result<usize> {
    let axes = match slot.kind {
        Kind::Bits { planes } => planes,
        Kind::Lookup {
            address_vars,
            packed: true,
        } if cfg!(feature = "bitz-lookup-bits") => address_vars,
        _ => 1 << (slot.vars() - slot.cell_vars),
    };
    (1usize << slot.cell_vars.max(ROW_VARS))
        .checked_mul(axes)
        .ok_or("advice layout overflow")
}

impl Plan {
    pub(super) fn new(layout: &Layout) -> Result<Self> {
        static GENERATOR: OnceLock<F128> = OnceLock::new();
        let slots = layout
            .slots
            .iter()
            .copied()
            .filter(|&slot| layout.active(slot))
            .map(|slot| Ok((slot.id, span(slot)?)))
            .collect::<Result<Vec<_>>>()?;
        let total = slots.iter().try_fold(0usize, |sum, &(_, size)| {
            sum.checked_add(size).ok_or("advice layout overflow")
        })?;
        let single = total
            .checked_next_power_of_two()
            .ok_or("advice layout overflow")?
            .ilog2() as usize;
        let mut capacity = vec![single.max(MIN_VARS)];
        let mut placement = vec![(0, 0); layout.slots.len()];
        let mut offset = 0;
        for &(id, size) in &slots {
            placement[id] = (0, offset);
            offset += size;
        }
        if cfg!(feature = "bitz-onehot-groups") && total > GROUP_BITS {
            let mut slots = slots;
            slots.sort_unstable_by_key(|&(id, size)| (std::cmp::Reverse(size), id));
            let mut groups: Vec<(usize, usize)> = Vec::new();
            for (id, size) in slots {
                let group = groups
                    .iter()
                    .enumerate()
                    .filter(|(_, &(capacity, used))| size <= capacity - used)
                    .min_by_key(|(_, &(capacity, used))| capacity - used)
                    .map(|(group, _)| group)
                    .unwrap_or_else(|| {
                        groups.push((size.next_power_of_two().max(GROUP_BITS), 0));
                        groups.len() - 1
                    });
                let used = &mut groups[group].1;
                placement[id] = (group, *used);
                *used += size;
            }
            capacity = groups
                .into_iter()
                .map(|(_, used)| (used.next_power_of_two().ilog2() as usize).max(MIN_VARS))
                .collect();
        } else if cfg!(feature = "bitz-lookup-groups") && single > MIN_VARS {
            let mut slots = slots;
            slots.sort_unstable_by_key(|&(id, size)| (std::cmp::Reverse(size), id));
            let large = 1usize << (single - 1);
            for small_vars in MIN_VARS..single - 1 {
                let caps = [large, 1 << small_vars];
                if total > caps.iter().sum() {
                    continue;
                }
                let mut used = [0; 2];
                let mut trial = placement.clone();
                let fits = slots.iter().all(|&(id, size)| {
                    let Some(group) = (0..2).find(|&i| size <= caps[i] - used[i]) else {
                        return false;
                    };
                    trial[id] = (group, used[group]);
                    used[group] += size;
                    true
                });
                if fits {
                    capacity = vec![single - 1, small_vars];
                    placement = trial;
                    break;
                }
            }
        }
        let groups = capacity
            .into_iter()
            .map(|vars| {
                let shape =
                    Shape::new(ROW_VARS, vars - ROW_VARS).map_err(|_| "invalid BitZ shape")?;
                let params = BitZParams::new(shape, *GENERATOR.get_or_init(smallest_generator))
                    .map_err(|_| "invalid BitZ parameters")?;
                let pcs = Pcs::new(&shape, SecurityLevel::Bits128)
                    .map_err(|_| "unsupported BitZ size")?;
                Ok((params, pcs))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            slots: placement,
            groups,
        })
    }

    pub(super) fn queries<F>(&self, openings: Vec<Opening<F>>) -> Vec<Vec<Opening<F>>> {
        let mut groups: Vec<_> = self.groups.iter().map(|_| Vec::new()).collect();
        for opening in openings {
            groups[self.slots[opening.slot].0].push(opening);
        }
        groups
    }
}

#[cfg(all(test, feature = "bitz-lookup-groups"))]
#[test]
fn routes_both_groups_through_the_verifier() {
    let mut layout = Layout::default();
    let mut requests = Vec::new();
    for vars in [MIN_VARS + 1, MIN_VARS] {
        let slot = layout.add(vars, Kind::Bits { planes: 1 }).unwrap();
        requests.push(Opening::new(
            slot,
            Claim {
                point: vec![F::zero(); vars],
                value: F::one(),
            },
        ));
    }
    let make_requests = || {
        requests
            .iter()
            .map(|q| Opening {
                slot: q.slot,
                claim: q.claim.clone(),
                planes: None,
            })
            .collect()
    };
    let plan = Plan::new(&layout).unwrap();
    assert_eq!(plan.groups.len(), 2);
    let mut advice = layout.witness();
    for slot in layout.slots.iter().copied() {
        let mut bits = vec![0; 1 << slot.cell_vars];
        bits[0] = 1;
        advice.put(slot, Part::Digits(bits.into())).unwrap();
    }
    let (roots, state) = BitzScheme::commit(&layout, &mut advice, &plan, &()).unwrap();
    let domain = jolt_transcript::protocol_tag(b"atlas/bitz-groups/test");
    let mut prover = prover_transcript::<Keccak>(&domain, b"");
    BitzScheme::bind(&roots, &mut prover);
    let proof = BitzScheme::open(
        &layout,
        make_requests(),
        &roots,
        state,
        &plan,
        &(),
        &mut prover,
    )
    .unwrap();
    let mut verifier = verifier_transcript::<Keccak>(&domain, b"", prover.narg_string());
    BitzScheme::bind(&roots, &mut verifier);
    BitzScheme::verify(
        &layout,
        make_requests(),
        &roots,
        &proof,
        &plan,
        &(),
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}
