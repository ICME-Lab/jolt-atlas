use super::GroupLayout;
use jolt_akita::AkitaScheduleArtifacts;
use jolt_openings::OpeningsError;
use std::collections::{BTreeMap, BTreeSet};

/// One standalone Akita commitment group, in the original polynomial order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedGroup {
    /// The shared polynomial representation and arity.
    pub layout: GroupLayout,
    /// Indices into the original list of polynomial layouts.
    pub polynomial_indices: Vec<usize>,
}

/// Partition ordered polynomial layouts into the fewest supported standalone groups.
///
/// The layout identifies the dense or one-hot family, one-hot column count and arity.
/// Polynomials keep their relative order within a layout. On ties, the larger group
/// goes first. Both sides use the same layout order and catalog to get the same plan.
pub fn plan_groups(
    layouts: &[GroupLayout],
    artifacts: &AkitaScheduleArtifacts,
) -> Result<Vec<PlannedGroup>, OpeningsError> {
    plan(layouts, artifacts, false)
}

/// Pad unsupported short domains and consider merging adjacent one-hot row domains.
///
/// A merge is selected when it produces fewer commitment groups or supplies a
/// grouping for a shape which otherwise lacks an exact partition.
pub fn plan_groups_with_padding(
    layouts: &[GroupLayout],
    artifacts: &AkitaScheduleArtifacts,
) -> Result<Vec<PlannedGroup>, OpeningsError> {
    plan(layouts, artifacts, true)
}

fn plan(
    layouts: &[GroupLayout],
    artifacts: &AkitaScheduleArtifacts,
    allow_padding: bool,
) -> Result<Vec<PlannedGroup>, OpeningsError> {
    let mut buckets = BTreeMap::<_, Vec<_>>::new();
    for (index, &layout) in layouts.iter().enumerate() {
        buckets.entry(layout).or_default().push(index);
    }

    let mut widths = BTreeMap::<GroupLayout, BTreeSet<usize>>::new();
    let families: BTreeSet<_> = buckets.keys().map(|layout| layout.one_hot_width).collect();
    for family in families {
        let catalog = match family {
            None => artifacts.dense_catalog(),
            Some(k) => artifacts.one_hot_catalog(k),
        }
        .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
        for row in catalog.rows() {
            let profiles = row.profiles();
            if profiles.precommitteds.is_empty() {
                let group = profiles.final_group.group;
                let layout = GroupLayout::new(group.num_vars(), family)?;
                if buckets.contains_key(&layout) || allow_padding {
                    widths
                        .entry(layout)
                        .or_default()
                        .insert(group.num_polynomials());
                }
            }
        }
    }

    if allow_padding {
        let layouts: Vec<_> = buckets.keys().copied().rev().collect();
        for pair in layouts.windows(2) {
            let [longer, shorter] = [pair[0], pair[1]];
            if longer.one_hot_width.is_none()
                || longer.one_hot_width != shorter.one_hot_width
                || longer.num_vars != shorter.num_vars + 1
            {
                continue;
            }
            let (Some(long), Some(short)) = (buckets.get(&longer), buckets.get(&shorter)) else {
                continue;
            };
            let Some(supported) = widths.get(&longer) else {
                continue;
            };
            let Some(merged) = partition(long.len() + short.len(), supported) else {
                continue;
            };
            let separate = partition(long.len(), supported).and_then(|long| {
                let short = partition(short.len(), widths.get(&shorter)?)?;
                Some(long.len() + short.len())
            });
            if separate.is_none_or(|count| merged.len() < count) {
                let short = buckets.remove(&shorter).unwrap();
                let long = buckets.get_mut(&longer).unwrap();
                long.extend(short);
                long.sort_unstable();
            }
        }
        pad_short_buckets(&mut buckets, &widths);
    }

    let mut groups = Vec::new();
    for (layout, indices) in buckets {
        let supported = widths.remove(&layout).unwrap_or_default();
        let counts = partition(indices.len(), &supported).ok_or_else(|| {
            OpeningsError::InvalidSetup(format!(
                "no exact Akita group partition for {} polynomials with {layout:?}; supported group sizes: {supported:?}",
                indices.len()
            ))
        })?;
        let mut remaining = indices.as_slice();
        for count in counts {
            let (group, rest) = remaining.split_at(count);
            groups.push(PlannedGroup {
                layout,
                polynomial_indices: group.to_vec(),
            });
            remaining = rest;
        }
    }
    Ok(groups)
}

fn pad_short_buckets(
    buckets: &mut BTreeMap<GroupLayout, Vec<usize>>,
    widths: &BTreeMap<GroupLayout, BTreeSet<usize>>,
) {
    let mut targets = BTreeMap::new();
    for (&layout, sizes) in widths {
        if sizes.contains(&1) {
            targets.entry(layout.one_hot_width).or_insert(layout);
        }
    }

    // Lift an entire original domain so its opening requests share the same padding.
    let moves: Vec<_> = buckets
        .iter()
        .filter_map(|(&layout, indices)| {
            let &target = targets.get(&layout.one_hot_width)?;
            (layout < target
                && widths
                    .get(&layout)
                    .and_then(|sizes| partition(indices.len(), sizes))
                    .is_none())
            .then_some((layout, target))
        })
        .collect();
    for (layout, target) in moves {
        let indices = buckets.remove(&layout).unwrap();
        buckets.entry(target).or_default().extend(indices);
    }
    for target in targets.values() {
        if let Some(indices) = buckets.get_mut(target) {
            indices.sort_unstable();
        }
    }
}

fn partition(count: usize, supported: &BTreeSet<usize>) -> Option<Vec<usize>> {
    let mut best = vec![None; count + 1];
    best[0] = Some((0, 0));
    for total in 1..=count {
        best[total] = supported
            .range(1..=total)
            .rev()
            .filter_map(|&size| best[total - size].map(|(groups, _)| (groups + 1, size)))
            .min_by_key(|&(groups, _)| groups);
    }
    let mut sizes = Vec::new();
    let mut remaining = count;
    while remaining != 0 {
        let (_, size) = best[remaining]?;
        sizes.push(size);
        remaining -= size;
    }
    Some(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::akita::schedule_artifacts;

    #[test]
    fn plans_exact_groups_for_mixed_layouts() {
        let dense = GroupLayout::new(14, None).unwrap();
        let k16 = GroupLayout::new(14, Some(16)).unwrap();
        let k256 = GroupLayout::new(14, Some(256)).unwrap();
        let longer = GroupLayout::new(15, Some(16)).unwrap();
        let layouts: Vec<_> = (0..10)
            .flat_map(|_| [dense, k16])
            .chain([k256, longer])
            .collect();
        let catalog = schedule_artifacts().unwrap();
        let groups = plan_groups(&layouts, &catalog).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|group| (group.layout, group.polynomial_indices.len()))
                .collect::<Vec<_>>(),
            vec![
                (dense, 8),
                (dense, 2),
                (k16, 2),
                (k16, 2),
                (k16, 2),
                (k16, 2),
                (k16, 2),
                (longer, 1),
                (k256, 1)
            ]
        );
        assert_eq!(
            groups[0].polynomial_indices,
            (0..16).step_by(2).collect::<Vec<_>>()
        );
    }

    #[test]
    fn plans_dense_remainders_without_padding() {
        let catalog = schedule_artifacts().unwrap();
        for num_vars in 11..=18 {
            let layout = GroupLayout::new(num_vars, None).unwrap();
            for count in 1_usize..=15 {
                let groups = plan_groups(&vec![layout; count], &catalog).unwrap();
                assert_eq!(
                    groups.len(),
                    count.count_ones() as usize,
                    "arity {num_vars}, {count} polynomials",
                );
                let indices: Vec<_> = groups
                    .into_iter()
                    .flat_map(|group| group.polynomial_indices)
                    .collect();
                assert_eq!(indices, (0..count).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn plans_smallest_dense_layout_in_groups_of_four_or_eight() {
        let catalog = schedule_artifacts().unwrap();
        let layout = GroupLayout::new(10, None).unwrap();
        let groups = plan_groups(&[layout; 12], &catalog).unwrap();
        assert_eq!(groups[0].polynomial_indices, (0..8).collect::<Vec<_>>());
        assert_eq!(groups[1].polynomial_indices, (8..12).collect::<Vec<_>>());
        assert!(plan_groups(&[layout; 9], &catalog).is_err());
        let exact = plan_groups_with_padding(&[layout; 12], &catalog).unwrap();
        assert_eq!(exact, groups);

        let short = GroupLayout::new(3, None).unwrap();
        let target = GroupLayout::new(11, None).unwrap();
        let layouts = [vec![layout; 9], vec![short, target]].concat();
        let padded = plan_groups_with_padding(&layouts, &catalog).unwrap();
        assert!(padded.iter().all(|group| group.layout == target));
        assert_eq!(
            padded
                .into_iter()
                .map(|group| group.polynomial_indices.len())
                .collect::<Vec<_>>(),
            [8, 2, 1]
        );
    }

    #[test]
    fn finds_exact_partition_when_largest_group_would_leave_an_unusable_remainder() {
        assert_eq!(partition(10, &BTreeSet::from([4, 5, 8])), Some(vec![5, 5]));
    }
}
