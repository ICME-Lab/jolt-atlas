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
        if total > GROUP_BITS {
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

#[cfg(test)]
#[test]
fn default_layout_keeps_a_large_slot_and_small_tail_separate() {
    let mut layout = Layout::default();
    let large = layout
        .add(GROUP_BITS.ilog2() as usize, Kind::Bits { planes: 1 })
        .unwrap();
    let tail = layout.add(10, Kind::Bits { planes: 1 }).unwrap();
    let plan = Plan::new(&layout).unwrap();
    assert_eq!(plan.slots[large.id], (0, 0));
    assert_eq!(plan.slots[tail.id], (1, 0));
    assert_eq!(
        plan.groups[0].0.shape().log_bits(),
        GROUP_BITS.ilog2() as usize
    );
    assert_eq!(plan.groups[1].0.shape().log_bits(), MIN_VARS);
}
