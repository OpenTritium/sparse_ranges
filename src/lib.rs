#![feature(btree_cursors)]
#![feature(if_let_guard)]
use std::{
    collections::{BTreeMap, BTreeSet},
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

#[derive(Debug, Default)]
pub struct OffsetRangeSet(BTreeSet<OffsetRange>);

impl OffsetRangeSet {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert_range(&mut self, rng: &OffsetRange) -> bool {
        let x = BTreeMap::<OffsetRange, usize>::new();

        // 定位光标到可能与 rng 相交的第一个位置
        let mut cursor = self
            .0
            .upper_bound_mut(Bound::Included(&OffsetRange::new(rng.start, rng.start)));
        if let Some(prev) = cursor.peek_prev()
            && prev.intersects_or_adjacent(rng)
        {
            cursor.prev();
        }
        // 检查是否为无操作（真包含或包含）
        // 我们只需要检查光标当前位置的下一个元素。
        // 如果这个元素（第一个可能与之相交的元素）已经包含了新范围，
        // 那么插入就是一个无操作，直接返回 false。
        if let Some(next) = cursor.peek_next()
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
            .is_some_and(|next| merged_rng.intersects_or_adjacent(next))
        {
            let range_to_merge = unsafe { cursor.remove_next().unwrap_unchecked() };
            merged_rng = unsafe { merged_rng.merge(&range_to_merge).unwrap_unchecked() };
        }
        unsafe { cursor.insert_after(merged_rng).unwrap_unchecked() };
        true
    }

    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut result_set = BTreeSet::new();
        let mut self_iter = self.0.iter().peekable();
        let mut other_iter = other.0.iter().peekable();

        // 'current_merged' 存储当前正在构建的、可能还会继续扩大的范围。
        let mut current_merged: Option<OffsetRange> = None;

        // 只要任一迭代器中还有元素，就继续循环。
        while self_iter.peek().is_some() || other_iter.peek().is_some() {
            // 从两个迭代器的头部选择 'start' 值最小的范围作为下一个处理对象。
            // 这里的 unwrap 是安全的，因为循环条件保证了至少有一个迭代器非空。
            let next_range = match (self_iter.peek(), other_iter.peek()) {
                (Some(&r1), Some(&r2)) => {
                    if r1.start() <= r2.start() {
                        unsafe { self_iter.next().unwrap_unchecked() }
                    } else {
                        unsafe { other_iter.next().unwrap_unchecked() }
                    }
                }
                (Some(_), None) => unsafe { self_iter.next().unwrap_unchecked() },
                (None, Some(_)) => unsafe { other_iter.next().unwrap_unchecked() },
                (None, None) => unreachable!(), // 循环条件已阻止此情况
            };

            match current_merged {
                None => {
                    // 如果没有正在进行的合并，那么 'next_range' 就是新的合并起点。
                    current_merged = Some(*next_range);
                }
                Some(merged) => {
                    // 如果存在一个正在合并的范围 'merged'，
                    // 检查 'next_range' 是否能与之合并。
                    if merged.intersects_or_adjacent(next_range) {
                        // 可以合并，则更新 'current_merged' 为合并后的更大范围。
                        // 这里的 unwrap 是安全的，因为上面的条件检查已经保证了可以合并。
                        current_merged = Some(merged.merge(next_range).unwrap());
                    } else {
                        // 如果不能合并（有间隙），说明 'merged' 这个范围已经构建完毕。
                        // 1. 将已完成的 'merged' 存入结果集。
                        result_set.insert(merged);
                        // 2. 将 'next_range' 作为新的合并起点。
                        current_merged = Some(*next_range);
                    }
                }
            }
        }

        // 循环结束后，最后一个正在进行的 'current_merged' 还没有被存入结果集。
        // 需要在这里把它加进去。
        if let Some(last_range) = current_merged {
            result_set.insert(last_range);
        }

        OffsetRangeSet(result_set)
    }
}

#[cfg(test)]
mod tests {
    use crate::{OffsetRange, OffsetRangeSet};
    use std::{collections::BTreeSet, ops::RangeInclusive};

    fn btree_set(ranges: &[RangeInclusive<usize>]) -> BTreeSet<OffsetRange> {
        ranges
            .iter()
            .map(|r| OffsetRange::new(*r.start(), *r.end()))
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
        assert_eq!(set1.union(&set2).0, expected.0);
    }

    #[test]
    fn test_union_with_empty_set() {
        let set1 = range_set(&[10..=20, 30..=40]);
        let set2 = range_set(&[]);
        let expected = range_set(&[10..=20, 30..=40]);

        // 验证操作的交换律
        assert_eq!(set1.union(&set2).0, expected.0);
        assert_eq!(set2.union(&set1).0, expected.0);
    }

    #[test]
    fn test_union_non_overlapping() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[30..=40]);
        let expected = range_set(&[10..=20, 30..=40]);
        assert_eq!(set1.union(&set2).0, expected.0);
    }

    #[test]
    fn test_union_interleaved_non_overlapping() {
        let set1 = range_set(&[10..=20, 50..=60]);
        let set2 = range_set(&[30..=40, 70..=80]);
        let expected = range_set(&[10..=20, 30..=40, 50..=60, 70..=80]);
        assert_eq!(set1.union(&set2).0, expected.0);
    }

    #[test]
    fn test_union_adjacent() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[21..=30]);
        let expected = range_set(&[10..=30]);
        assert_eq!(set1.union(&set2).0, expected.0);
    }

    #[test]
    fn test_union_simple_overlap() {
        let set1 = range_set(&[10..=20]);
        let set2 = range_set(&[15..=25]);
        let expected = range_set(&[10..=25]);
        assert_eq!(set1.union(&set2).0, expected.0);
    }

    #[test]
    fn test_union_one_contains_another() {
        let set1 = range_set(&[10..=100]);
        let set2 = range_set(&[20..=30]);
        let expected = range_set(&[10..=100]);
        assert_eq!(set1.union(&set2).0, expected.0);
        assert_eq!(set2.union(&set1).0, expected.0);
    }

    #[test]
    fn test_union_identical_sets() {
        let set1 = range_set(&[10..=20, 30..=40]);
        let set2 = range_set(&[10..=20, 30..=40]);
        let expected = range_set(&[10..=20, 30..=40]);
        assert_eq!(set1.union(&set2).0, expected.0);
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
        assert_eq!(set1.union(&set2).0, expected.0);
        assert_eq!(set2.union(&set1).0, expected.0);
    }

    #[test]
    fn test_union_one_range_swallows_many() {
        let set1 = range_set(&[0..=100]);
        let set2 = range_set(&[10..=20, 30..=40, 50..=60]);
        let expected = range_set(&[0..=100]);
        assert_eq!(set1.union(&set2).0, expected.0);
        assert_eq!(set2.union(&set1).0, expected.0);
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
        assert_eq!(set1.union(&set2).0, expected.0);
        assert_eq!(set2.union(&set1).0, expected.0);
    }
}
