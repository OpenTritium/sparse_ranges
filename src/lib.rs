#![feature(btree_cursors)]
#![feature(if_let_guard)]
use std::{
    collections::BTreeMap,
    ops::{self, Bound},
};

/// inclusive
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
pub struct OffsetRange {
    start: usize,
    last: usize,
}
impl OffsetRange {
    #[inline]
    pub fn new(start: usize, last: usize) -> Self {
        debug_assert!(start <= last);
        OffsetRange { start, last }
    }

    #[inline]
    pub fn start(&self) -> usize {
        self.start
    }

    #[inline]
    pub fn last(&self) -> usize {
        self.last
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.last - self.start + 1
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    pub fn contains_n(&self, n: usize) -> bool {
        self.start <= n && n <= self.last
    }

    /// 检查此范围是否 **完全包含** 另一个范围 `rhs`。
    ///
    /// 这是一个严格的子集关系。如果 `self.contains_range(rhs)` 为 true,
    /// 那么 `rhs` 中的所有点都必须在 `self` 中。
    ///
    /// 示例: `[10, 30].contains_range(&[15, 25])` -> `true`
    ///       `[10, 20].contains_range(&[15, 25])` -> `false`
    #[inline]
    fn contains(&self, other: &Self) -> bool {
        self.start <= other.start && self.last >= other.last
    }

    /// 检查此范围是否与另一个范围 `other` **有任何重叠**。
    ///
    /// 只要两个范围有至少一个共同点（包括边缘接触），此函数就返回 true。
    /// 这是一个比 `contains_range` 更宽泛的检查。
    ///
    /// 示例: `[10, 20].intersects(&[15, 25])` -> `true`
    ///       `[10, 20].intersects(&[20, 30])` -> `true` (边缘接触)
    ///       `[10, 20].intersects(&[30, 40])` -> `false`
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.start <= other.last && self.last >= other.start
    }

    #[inline]
    fn intersects_or_adjacent(&self, other: &Self) -> bool {
        self.start <= other.last + 1 && self.last + 1 >= other.start
    }

    /// 如果两个范围相交，则将它们合并成一个能覆盖两者的最小范围。
    ///
    /// 如果不相交，则返回 `None`。
    /// 此函数依赖 `intersects()` 的结果。
    #[inline]
    pub fn merge(&self, other: &Self) -> Option<Self> {
        self.intersects_or_adjacent(other).then_some({
            let start = self.start.min(other.start);
            let last = self.last.max(other.last);
            OffsetRange::new(start, last)
        })
    }
}

// todo range版本转换，需要处理不包含错误比如  0..0
impl From<&ops::RangeInclusive<usize>> for OffsetRange {
    fn from(rng: &ops::RangeInclusive<usize>) -> Self {
        Self {
            start: *rng.start(),
            last: *rng.end(),
        }
    }
}

impl From<(usize, usize)> for OffsetRange {
    #[inline]
    fn from(rng: (usize, usize)) -> Self {
        debug_assert!(rng.0 <= rng.1);
        Self {
            start: rng.0,
            last: rng.1,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OffsetRangeSet(BTreeMap<usize, usize>);

impl OffsetRangeSet {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert_range(&mut self, rng: &OffsetRange) -> bool {
        // 定位光标到可能与 rng 相交的第一个位置
        let mut cursor = self.0.upper_bound_mut(Bound::Included(&rng.start));
        if let Some(prev) = cursor.peek_prev().map(|(l, r)| OffsetRange::from((*l, *r)))
            && prev.intersects_or_adjacent(rng)
        {
            cursor.prev();
        }
        // 检查是否为无操作（真包含或包含）
        // 我们只需要检查光标当前位置的下一个元素。
        // 如果这个元素（第一个可能与之相交的元素）已经包含了新范围，
        // 那么插入就是一个无操作，直接返回 false。
        if let Some(next) = cursor.peek_next().map(|(l, r)| OffsetRange::from((*l, *r)))
            && next.contains(rng)
        {
            return false;
        }
        // 如果不是无操作，则执行合并/插入逻辑
        // 因为我们已经排除了被完全包含的情况，任何后续操作都必然会修改集合
        let mut merged_rng = *rng;

        // 只要下一个元素存在(peek_next)并且(map_or)与我们的范围相交，就继续循环
        while cursor
            .peek_next()
            .map(|(l, r)| OffsetRange::from((*l, *r)))
            .is_some_and(|next| merged_rng.intersects_or_adjacent(&next))
        {
            let range_to_merge: OffsetRange =
                unsafe { cursor.remove_next().unwrap_unchecked().into() };
            merged_rng = unsafe { merged_rng.merge(&range_to_merge).unwrap_unchecked() };
        }
        unsafe {
            cursor
                .insert_after(merged_rng.start, merged_rng.last)
                .unwrap_unchecked()
        };
        true
    }

    #[must_use]
    pub fn union_merge(&self, other: &Self) -> Self {
        let mut result_set = BTreeMap::new();
        let mut self_iter = self.0.iter().peekable();
        let mut other_iter = other.0.iter().peekable();

        // 存储当前正在构建的、可能还会继续扩大的范围。
        let mut current_merged: Option<OffsetRange> = None;

        // 使用无限循环，并在内部处理所有情况，包括终止条件。
        loop {
            // 从两个迭代器的头部选择 'start' 值最小的范围作为下一个处理对象。
            // 这个 match 结构清晰地处理了所有情况，并自然地包含了循环的退出点。
            let next_range_tuple = unsafe {
                match (self_iter.peek(), other_iter.peek()) {
                    // 两个迭代器都有元素，选择起始点更早的那个。
                    (Some(&s_rng), Some(&o_rng)) => {
                        if s_rng.0 <= o_rng.0 {
                            self_iter.next().unwrap_unchecked() // 安全：因为 peek() 返回 Some
                        } else {
                            other_iter.next().unwrap_unchecked()
                        }
                    }
                    // 只有 self_iter 有元素，取之。
                    (Some(_), None) => self_iter.next().unwrap_unchecked(),
                    // 只有 other_iter 有元素，取之。
                    (None, Some(_)) => other_iter.next().unwrap_unchecked(),
                    // 两个迭代器都已耗尽，合并过程结束。
                    (None, None) => break,
                }
            };

            // 将元组转换为我们的范围类型
            let next_range = OffsetRange::new(*next_range_tuple.0, *next_range_tuple.1);

            match current_merged.as_mut() {
                // Case 1: 这是第一个范围，或者我们刚完成了一个合并间隙。
                // 直接将 next_range 作为新的合并起点。
                None => {
                    current_merged = Some(next_range);
                }
                // Case 2: 当前有一个正在合并的范围。
                Some(merged) if merged.intersects_or_adjacent(&next_range) => {
                    // 新范围与当前合并的范围重叠或相邻，扩大 `merged` 的边界。
                    // 直接修改，因为 `as_mut()` 提供了可变引用。
                    merged.last = merged.last.max(next_range.last);
                }
                Some(merged) => {
                    // 新范围与当前合并的范围有间隙。
                    // 1. 说明 `merged` 已经构建完毕，将其存入结果集。
                    result_set.insert(merged.start, merged.last);
                    // 2. 将 `next_range` 作为新的合并起点。
                    *merged = next_range;
                }
            }
        }
        // 循环结束后，最后一个正在进行的 `current_merged` 还没有被存入结果集。
        if let Some(last_rng) = current_merged {
            result_set.insert(last_rng.start, last_rng.last);
        }

        OffsetRangeSet(result_set)
    }

    #[must_use]
    #[inline]
    fn union(&self, other: &Self) -> Self {
        if self.0.is_empty() {
            return other.clone();
        }
        if other.0.is_empty() {
            return self.clone();
        }
        let self_ranges_count = self.0.len();
        let other_ranges_count = other.0.len();
        let insertion_cost_estimate = other_ranges_count * self_ranges_count.ilog2() as usize;
        let merge_cost_estimate = self_ranges_count + other_ranges_count;
        if insertion_cost_estimate < merge_cost_estimate && other_ranges_count < self_ranges_count {
            let mut result = self.0.clone();
            for (start, last) in &other.0 {
                result.insert(*start, *last);
            }
            OffsetRangeSet(result)
        } else {
            let result = self.clone();
            result.union_merge(other)
        }
    }
    pub fn difference(&self, other: &Self) -> Self {
        if self.0.is_empty() || other.0.is_empty() {
            return self.clone();
        }

        let mut result = OffsetRangeSet::new();
        let mut a_iter = self.0.iter();
        let mut b_iter = other.0.iter().peekable();

        // 从 A 中获取第一个范围
        let mut current_a = match a_iter.next() {
            Some((&start, &last)) => OffsetRange::new(start, last),
            None => return result, // A 是空的
        };

        loop {
            // 查看 B 的下一个范围
            match b_iter.peek() {
                // Case 1: B 中还有范围
                Some(&(&b_start, &b_last)) => {
                    let b_range = OffsetRange::new(b_start, b_last);

                    // 如果 b_range 完全在 current_a 之前，跳过这个 b_range
                    if b_range.last() < current_a.start() {
                        b_iter.next(); // 消耗掉 b_range
                        continue;
                    }

                    // 如果 b_range 完全在 current_a 之后，说明 current_a 不会再被裁剪
                    // 完成对 current_a 的处理，然后尝试从 A 获取下一个
                    if b_range.start() > current_a.last() {
                        result.insert_range(&current_a);
                        if let Some((&s, &l)) = a_iter.next() {
                            current_a = OffsetRange::new(s, l);
                            continue;
                        } else {
                            break;
                        }
                    }

                    // --- 处理重叠 ---

                    // 如果 b_range 在 current_a 的前面留下了一部分
                    if b_range.start() > current_a.start() {
                        let prefix = OffsetRange::new(current_a.start(), b_range.start() - 1);
                        result.insert_range(&prefix);
                    }

                    // 更新 current_a 的起点，跳过被 b_range 覆盖的部分
                    // 如果 b_range.last() 溢出了，说明 current_a 被完全覆盖
                    if let Some(new_start) = b_range.last().checked_add(1) {
                        // 如果 new_start 已经超出了 current_a 的范围
                        if new_start > current_a.last() {
                            // current_a 被完全处理完毕，获取下一个
                            current_a = match a_iter.next() {
                                Some((&s, &l)) => OffsetRange::new(s, l),
                                None => break, // A 耗尽，结束
                            };
                        } else {
                            // current_a 还有剩余，更新起点继续处理
                            current_a = OffsetRange::new(new_start, current_a.last());
                        }
                    } else {
                        // b_range.last() 是 usize::MAX，current_a 之后不可能还有剩余
                        current_a = match a_iter.next() {
                            Some((&s, &l)) => OffsetRange::new(s, l),
                            None => break, // A 耗尽，结束
                        };
                    }
                }
                // Case 2: B 已经耗尽，A 中所有剩余的范围都属于结果
                None => {
                    result.insert_range(&current_a);
                    for (&start, &last) in a_iter {
                        result.insert_range(&OffsetRange::new(start, last));
                    }
                    break; // 结束主循环
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{OffsetRange, OffsetRangeSet};
    use std::{collections::BTreeMap, ops::RangeInclusive};

    fn btree_set(ranges: &[RangeInclusive<usize>]) -> BTreeMap<usize, usize> {
        ranges
            .iter()
            .map(|rng| (*rng.start(), *rng.end()))
            .collect()
    }

    #[test]
    fn test_insert_into_empty_set() {
        let mut set = OffsetRangeSet::new();
        let inserted = set.insert_range(&OffsetRange::new(10, 20));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=20)]));
    }

    #[test]
    fn test_insert_non_overlapping_before() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(0, 5));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(0..=5), (10..=20)]));
    }

    #[test]
    fn test_insert_non_overlapping_after() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(25, 30));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=20), (25..=30)]));
    }

    #[test]
    fn test_insert_non_overlapping_between() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20), (30..=40)]);
        let inserted = set.insert_range(&OffsetRange::new(22, 28));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=20), (22..=28), (30..=40)]));
    }

    #[test]
    fn test_insert_overlapping_end() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(15, 25));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=25)]));
    }

    #[test]
    fn test_insert_overlapping_start() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(5, 15));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(5..=20)]));
    }

    #[test]
    fn test_insert_merging_two_ranges() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20), (30..=40)]);
        let inserted = set.insert_range(&OffsetRange::new(15, 35));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=40)]));
    }

    #[test]
    fn test_insert_merging_multiple_ranges() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20), (30..=40), (50..=60)]);
        let inserted = set.insert_range(&OffsetRange::new(15, 55));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=60)]));
    }

    #[test]
    fn test_insert_fully_contained() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=100)]);
        let inserted = set.insert_range(&OffsetRange::new(20, 30));
        assert!(!inserted, "Should not insert a fully contained range");
        assert_eq!(set.0, btree_set(&[(10..=100)]));
    }

    #[test]
    fn test_insert_fully_containing_one_range() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(20..=30)]);
        let inserted = set.insert_range(&OffsetRange::new(10, 40));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=40)]));
    }

    #[test]
    fn test_insert_fully_containing_multiple_ranges() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(20..=30), (40..=50)]);
        let inserted = set.insert_range(&OffsetRange::new(10, 60));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=60)]));
    }

    #[test]
    fn test_insert_adjacent_before() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        // `intersects` 是包含的，所以 `[5,9]` 和 `[10,20]` 不相交
        let inserted = set.insert_range(&OffsetRange::new(5, 9));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(5..=20)]));
    }

    #[test]
    fn test_insert_adjacent_after() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(21, 25));
        assert!(inserted);
        assert_eq!(set.0, btree_set(&[(10..=25)]));
    }

    #[test]
    fn test_insert_duplicate_range() {
        let mut set = OffsetRangeSet::new();
        set.0 = btree_set(&[(10..=20)]);
        let inserted = set.insert_range(&OffsetRange::new(10, 20));
        assert!(!inserted, "Should not insert a duplicate range");
        assert_eq!(set.0, btree_set(&[(10..=20)]));
    }

    fn range_set(ranges: &[RangeInclusive<usize>]) -> OffsetRangeSet {
        OffsetRangeSet(btree_set(ranges))
    }

    #[test]
    fn test_union_both_empty() {
        let set1 = range_set(&[]);
        let set2 = range_set(&[]);
        let expected = range_set(&[]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_with_empty_set() {
        let set1 = range_set(&[10..=20, 30..=40]);
        let set2 = range_set(&[]);
        let expected = range_set(&[10..=20, 30..=40]);

        // 验证操作的交换律
        assert_eq!(set1.union_merge(&set2).0, expected.0);
        assert_eq!(set2.union_merge(&set1).0, expected.0);
    }

    #[test]
    fn test_union_non_overlapping() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[30..=40]);
        let expected = range_set(&[10..=20, 30..=40]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_interleaved_non_overlapping() {
        let set1 = range_set(&[10..=20, 50..=60]);
        let set2 = range_set(&[30..=40, 70..=80]);
        let expected = range_set(&[10..=20, 30..=40, 50..=60, 70..=80]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_adjacent() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[21..=30]);
        let expected = range_set(&[10..=30]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_simple_overlap() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[15..=25]);
        let expected = range_set(&[10..=25]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_one_contains_another() {
        let set1 = range_set(&[10..=100]);
        let set2 = range_set(&[20..=30]);
        let expected = range_set(&[10..=100]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
        assert_eq!(set2.union_merge(&set1).0, expected.0);
    }

    #[test]
    fn test_union_identical_sets() {
        let set1 = range_set(&[10..=20, 30..=40]);
        let set2 = range_set(&[10..=20, 30..=40]);
        let expected = range_set(&[10..=20, 30..=40]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
    }

    #[test]
    fn test_union_complex_merge_and_gaps() {
        let set1 = range_set(&[10..=20, 30..=40, 60..=70]);
        let set2 = range_set(&[15..=35, 65..=75]);
        // 预期合并过程:
        // [10,20] 和 [15,35] -> [10,35]
        // [10,35] 和 [30,40] -> [10,40]
        // [60,70] 和 [65,75] -> [60,75]
        let expected = range_set(&[10..=40, 60..=75]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
        assert_eq!(set2.union_merge(&set1).0, expected.0);
    }

    #[test]
    fn test_union_one_range_swallows_many() {
        let set1 = range_set(&[0..=100]);
        let set2 = range_set(&[10..=20, 30..=40, 50..=60]);
        let expected = range_set(&[0..=100]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
        assert_eq!(set2.union_merge(&set1).0, expected.0);
    }

    #[test]
    fn test_union_multiple_merges_from_both_sides() {
        let set1 = range_set(&[0..=10, 20..=30, 40..=50]);
        let set2 = range_set(&[5..=25, 35..=45]);
        // 预期合并过程:
        // [0,10] 和 [5,25] -> [0,25]
        // [0,25] 和 [20,30] -> [0,30]
        // [35,45] 和 [40,50] -> [35,50]
        let expected = range_set(&[0..=30, 35..=50]);
        assert_eq!(set1.union_merge(&set2).0, expected.0);
        assert_eq!(set2.union_merge(&set1).0, expected.0);
    }

    #[test]
    fn test_difference_empty() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[10..=20]));
        assert_eq!(set_b.difference(&set_a).0, btree_set(&[]));
    }

    #[test]
    fn test_difference_non_overlapping() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[30..=40]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[10..=20]));
        assert_eq!(set_b.difference(&set_a).0, btree_set(&[30..=40]));
    }

    #[test]
    fn test_difference_b_carves_start() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[5..=15]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[16..=20]));
    }

    #[test]
    fn test_difference_b_carves_end() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[15..=25]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[10..=14]));
    }

    #[test]
    fn test_difference_b_splits_a() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[13..=17]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[10..=12, 18..=20]));
    }

    #[test]
    fn test_difference_b_contains_a() {
        let set_a = range_set(&[10..=20]);
        let set_b = range_set(&[5..=25]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[]));
    }

    #[test]
    fn test_difference_a_contains_b() {
        let set_a = range_set(&[0..=100]);
        let set_b = range_set(&[20..=30]);
        assert_eq!(set_a.difference(&set_b).0, btree_set(&[0..=19, 31..=100]));
    }

    #[test]
    fn test_difference_multiple_holes() {
        let set_a = range_set(&[0..=100]);
        let set_b = range_set(&[10..=20, 40..=50, 80..=90]);
        let expected = btree_set(&[0..=9, 21..=39, 51..=79, 91..=100]);
        assert_eq!(set_a.difference(&set_b).0, expected);
    }

    #[test]
    fn test_difference_b_merges_and_carves() {
        let set_a = range_set(&[0..=50, 60..=100]);
        let set_b = range_set(&[40..=70]); // This range in B bridges the gap in A
        let expected = btree_set(&[0..=39, 71..=100]);
        assert_eq!(set_a.difference(&set_b).0, expected);
    }
}
