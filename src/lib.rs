#![feature(btree_cursors)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![cfg_attr(feature = "serde", allow(clippy::unsafe_derive_deserialize))]

use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Display},
    ops::{self, BitOr, BitOrAssign, Bound, Deref, Not, Sub, SubAssign},
};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An inclusive range defined by start and last offsets.
///
/// This struct represents a contiguous range of unsigned integers where both
/// the start and end points are included in the range. It provides various
/// utility methods for manipulating and querying ranges.
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Range {
    start: usize,
    last: usize,
}

impl Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}..={}", self.start, self.last) }
}

impl Range {
    /// Creates a new range with the given start and last values.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting offset (inclusive)
    /// * `last` - The ending offset (inclusive)
    ///
    /// # Panics
    ///
    /// Panics if `last >= usize::MAX` or if `start > last`.
    #[must_use]
    #[inline]
    pub fn new(start: usize, last: usize) -> Self {
        assert!(last < usize::MAX, "last must be less than usize::MAX");
        assert!(start <= last, "start must be less than or equal to last");
        Self { start, last }
    }

    #[must_use]
    #[inline]
    /// Creates a new range without checking if start <= last.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `start` is less than or equal to `last`.
    pub const unsafe fn new_unchecked(start: usize, last: usize) -> Self {
        debug_assert!(last < usize::MAX, "last must be less than usize::MAX");
        debug_assert!(start <= last, "start must be less than or equal to last");
        Self { start, last }
    }

    /// Returns the start offset of the range.
    #[inline]
    #[must_use]
    pub const fn start(&self) -> usize { self.start }

    /// Returns the last offset of the range.
    #[inline]
    #[must_use]
    pub const fn last(&self) -> usize { self.last }

    /// Returns the length of the range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range = Range::new(5, 10);
    /// assert_eq!(range.len(), 6);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        debug_assert!(self.start <= self.last);
        debug_assert!(self.last < usize::MAX);
        self.last - self.start + 1
    }

    /// Always returns `false` because a `Range` is never empty.
    ///
    /// A `Range` is always considered non-empty because it represents
    /// an inclusive range from start to last where both ends are included.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool { false }

    /// Checks if the range contains a specific offset.
    ///
    /// # Arguments
    ///
    /// * `n` - The offset to check
    ///
    /// # Returns
    ///
    /// `true` if `n` is within the range (inclusive), `false` otherwise.
    #[inline]
    #[must_use]
    pub const fn contains_n(&self, n: usize) -> bool { self.start <= n && n <= self.last }

    /// Checks if the range contains another range.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to check for containment
    ///
    /// # Returns
    ///
    /// `true` if `other.start..=other.last` is completely within `self.start..=self.last`.
    #[inline]
    #[must_use]
    pub const fn contains(&self, other: &Self) -> bool { self.start <= other.start && self.last >= other.last }

    /// Checks if two ranges intersect.
    ///
    /// Two ranges intersect if they share at least one common point.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to check for intersection
    ///
    /// # Returns
    ///
    /// `true` if the ranges intersect, `false` otherwise.
    #[inline]
    #[must_use]
    pub const fn intersects(&self, other: &Self) -> bool { self.start <= other.last && self.last >= other.start }

    /// Checks if two ranges intersect or are adjacent.
    ///
    /// Ranges are considered adjacent if one ends exactly where the other begins.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to check
    ///
    /// # Returns
    ///
    /// `true` if the ranges intersect or are adjacent, `false` otherwise.
    #[inline]
    #[must_use]
    const fn intersects_or_adjacent(&self, other: &Self) -> bool {
        self.start.saturating_sub(1) <= other.last && other.start.saturating_sub(1) <= self.last
    }

    /// Checks if two ranges are adjacent.
    ///
    /// Two ranges are adjacent if one ends exactly where the other begins.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to check for adjacency
    ///
    /// # Returns
    ///
    /// `true` if the ranges are adjacent, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range1 = Range::new(0, 5);
    /// let range2 = Range::new(6, 10);
    /// assert!(range1.is_adjacent(&range2));
    ///
    /// let range3 = Range::new(0, 5);
    /// let range4 = Range::new(7, 10);
    /// assert!(!range3.is_adjacent(&range4));
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_adjacent(&self, other: &Self) -> bool {
        (self.last < usize::MAX && self.last + 1 == other.start)
            || (other.last < usize::MAX && other.last + 1 == self.start)
    }

    /// Returns the midpoint of the range.
    ///
    /// The midpoint is calculated as the average of start and last, rounded down.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range = Range::new(0, 10);
    /// assert_eq!(range.midpoint(), 5);
    ///
    /// let range = Range::new(5, 8);
    /// assert_eq!(range.midpoint(), 6);
    /// ```
    #[inline]
    #[must_use]
    pub const fn midpoint(&self) -> usize { self.start + (self.last - self.start) / 2 }

    /// Returns the intersection of two ranges.
    ///
    /// If the ranges do not intersect, returns `None`.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to intersect with
    ///
    /// # Returns
    ///
    /// `Some(Range)` containing the overlapping part, or `None` if no intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range1 = Range::new(0, 10);
    /// let range2 = Range::new(5, 15);
    /// let intersection = range1.intersection(&range2).unwrap();
    /// assert_eq!(intersection, Range::new(5, 10));
    ///
    /// let range3 = Range::new(20, 30);
    /// assert!(range1.intersection(&range3).is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        self.intersects(other).then(|| {
            let start = self.start.max(other.start);
            let last = self.last.min(other.last);
            Self::new(start, last)
        })
    }

    /// Returns the difference between two ranges.
    ///
    /// The difference is the part of `self` that is not covered by `other`.
    /// Returns a tuple of optional ranges representing the left and right parts.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to subtract
    ///
    /// # Returns
    ///
    /// A tuple `(left, right)` where:
    /// - `left` is the part of `self` before `other` (if any)
    /// - `right` is the part of `self` after `other` (if any)
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range1 = Range::new(0, 10);
    /// let range2 = Range::new(3, 7);
    /// let (left, right) = range1.difference(&range2);
    /// assert_eq!(left, Some(Range::new(0, 2)));
    /// assert_eq!(right, Some(Range::new(8, 10)));
    ///
    /// let range3 = Range::new(0, 5);
    /// let range4 = Range::new(2, 8);
    /// let (left2, right2) = range3.difference(&range4);
    /// assert_eq!(left2, Some(Range::new(0, 1)));
    /// assert_eq!(right2, None);
    ///
    /// let range5 = Range::new(0, 10);
    /// let range6 = Range::new(0, 10);
    /// let (left3, right3) = range5.difference(&range6);
    /// assert_eq!(left3, None);
    /// assert_eq!(right3, None);
    /// ```
    #[inline]
    #[must_use]
    pub fn difference(&self, other: &Self) -> (Option<Self>, Option<Self>) {
        if !self.intersects(other) {
            return (Some(*self), None);
        }
        let left = (self.start < other.start && other.start > 0).then(|| Self::new(self.start, other.start - 1));
        let right = (self.last > other.last && other.last < usize::MAX).then(|| Self::new(other.last + 1, self.last));
        (left, right)
    }

    /// Attempts to merge two ranges.
    ///
    /// If the ranges intersect or are adjacent, returns a new range that covers both.
    /// Otherwise, returns `None`.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to merge with
    ///
    /// # Returns
    ///
    /// `Some(range)` with the merged range if successful, `None` otherwise.
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> Option<Self> {
        self.intersects_or_adjacent(other).then_some({
            let start = self.start.min(other.start);
            let last = self.last.max(other.last);
            Self::new(start, last)
        })
    }
}

#[cfg(feature = "http")]
impl Range {
    /// Converts the range to an HTTP Range header string format.
    ///
    /// Returns a string in the format "start-last" suitable for use in
    /// HTTP Range headers.
    ///
    /// # Returns
    ///
    /// A string representation of the range in HTTP Range header format.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range = Range::new(0, 499);
    /// assert_eq!(range.to_http_range_header(), "0-499");
    /// ```
    #[inline]
    #[must_use]
    pub fn to_http_range_header(&self) -> String { format!("{}-{}", self.start, self.last) }
}

impl TryFrom<&ops::Range<usize>> for Range {
    type Error = Error;

    /// Attempts to create a `Range` from a standard library range.
    ///
    /// This conversion takes a half-open range (`start..end`) and converts it
    /// to an inclusive range (`start..=last`).
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to convert
    ///
    /// # Errors
    ///
    /// Returns `Error::IndexOverflow` if the range end is 0 (would underflow when
    /// computing `end - 1`), or if the resulting range would be invalid (start > last
    /// or last >= `usize::MAX`).
    #[inline]
    fn try_from(rng: &ops::Range<usize>) -> Result<Self, Self::Error> {
        let start = rng.start;
        let last = rng.end.checked_sub(1).ok_or(Error::IndexOverflow)?;
        Ok(Self::new(start, last))
    }
}

impl From<&ops::RangeInclusive<usize>> for Range {
    /// Creates a `Range` from a reference to an inclusive range.
    ///
    /// # Panics
    ///
    /// This function does not validate the range. The caller must ensure that the range
    /// end value is less than `usize::MAX`. If the range from the standard library is
    /// invalid, this may create an invalid `Range`.
    ///
    /// # Arguments
    ///
    /// * `rng` - The inclusive range to convert
    #[inline]
    fn from(rng: &ops::RangeInclusive<usize>) -> Self { Self { start: *rng.start(), last: *rng.end() } }
}

impl From<(usize, usize)> for Range {
    /// Creates a `Range` from a tuple of (start, last).
    ///
    /// # Arguments
    ///
    /// * `rng` - A tuple where the first element is the start and the second is the last
    ///
    /// # Panics
    ///
    /// Panics if the start value is greater than the last value, or if the last value
    /// is greater than or equal to `usize::MAX`.
    #[inline]
    fn from((start, last): (usize, usize)) -> Self { Self::new(start, last) }
}

impl PartialEq<Range> for RangeSet {
    /// Checks if a `RangeSet` is equal to a single `Range`.
    ///
    /// This method returns `true` if the `RangeSet` contains exactly one range
    /// and that range is equal to the provided `Range`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let range = Range::new(0, 10);
    /// let mut set = RangeSet::new();
    /// set.insert_range(&range);
    /// assert_eq!(set, range);
    ///
    /// let mut set2 = RangeSet::new();
    /// set2.insert_range(&Range::new(0, 5));
    /// set2.insert_range(&Range::new(7, 10)); // Note: gap between 5 and 7
    /// assert_ne!(set2, range); // set2 has two ranges
    /// ```
    #[inline]
    fn eq(&self, other: &Range) -> bool {
        if self.ranges_count() != 1 {
            return false;
        }
        let (&start, &last) = unsafe { self.0.first_key_value().unwrap_unchecked() };
        start == other.start() && last == other.last()
    }
}

impl PartialEq<RangeSet> for Range {
    /// Checks if a `Range` is equal to a `RangeSet`.
    ///
    /// This method delegates to the `PartialEq<Range>` implementation for `RangeSet`,
    /// so it returns `true` if the `RangeSet` contains exactly one range and that range
    /// is equal to this `Range`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let range = Range::new(0, 10);
    /// let mut set = RangeSet::new();
    /// set.insert_range(&range);
    /// assert_eq!(range, set);
    /// ```
    #[inline]
    fn eq(&self, other: &RangeSet) -> bool { other.eq(self) }
}

impl From<Range> for RangeSet {
    /// Creates a `RangeSet` containing a single range.
    ///
    /// This is equivalent to calling `insert_range` on an empty set.
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to include in the set
    #[inline]
    fn from(rng: Range) -> Self { Self(BTreeMap::from([(rng.start, rng.last)])) }
}

/// A set of non-overlapping inclusive ranges.
///
/// This data structure efficiently maintains a set of non-overlapping,
/// inclusive ranges of unsigned integers. It automatically merges overlapping
/// or adjacent ranges when inserting new ranges.
///
/// # Examples
///
/// ```
/// # use sparse_ranges::{Range, RangeSet};
/// let mut set = RangeSet::new();
/// set.insert_range(&Range::new(0, 5));
/// set.insert_range(&Range::new(10, 15));
/// // The set now contains two separate ranges
/// ```
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RangeSet(BTreeMap<usize, usize>);

impl Debug for RangeSet {
    /// Formats the `RangeSet` for debugging purposes.
    ///
    /// This implementation displays the range set in a human-readable format,
    /// showing all the ranges contained in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// set.insert_range(&Range::new(20, 30));
    /// assert_eq!(format!("{:?}", set), "RangeSet {0..=10, 20..=30}");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeSet ")?;
        let mut set = f.debug_set();
        for (&start, &last) in &self.0 {
            set.entry(&format_args!("{start}..={last}"));
        }
        set.finish()
    }
}

impl RangeSet {
    /// Creates a new, empty `RangeSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::RangeSet;
    /// let set = RangeSet::new();
    /// assert!(set.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Returns the total number of offsets covered by all ranges in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 5));  // 6 offsets
    /// set.insert_range(&Range::new(10, 12)); // 3 offsets
    /// assert_eq!(set.len(), 9);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize { self.0.iter().map(|(start, last)| last - start + 1).sum() }

    /// Returns the number of ranges in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 5));
    /// set.insert_range(&Range::new(10, 12));
    /// assert_eq!(set.ranges_count(), 2);
    #[inline]
    #[must_use]
    pub fn ranges_count(&self) -> usize { self.0.len() }

    /// Checks if the set is empty.
    ///
    /// # Returns
    ///
    /// `true` if the set contains no ranges, `false` otherwise.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Returns the start offset of the first range in the set.
    ///
    /// # Returns
    ///
    /// The start offset of the first range, or `None` if the set is empty.
    #[inline]
    #[must_use]
    pub fn start(&self) -> Option<usize> { self.0.first_key_value().map(|(start, _)| *start) }

    /// Returns the end offset of the last range in the set.
    ///
    /// # Returns
    ///
    /// The last offset of the last range, or `None` if the set is empty.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<usize> { self.0.last_key_value().map(|(_, last)| *last) }

    /// Checks if the set contains a specific offset.
    ///
    /// # Arguments
    ///
    /// * `n` - The offset to check for containment
    ///
    /// # Returns
    ///
    /// `true` if the offset is covered by any range in the set, `false` otherwise.
    #[inline]
    #[must_use]
    pub fn contains_n(&self, n: usize) -> bool {
        if let Some((_, last)) = self.0.range(..=n).next_back() {
            return n <= *last;
        }
        false
    }

    /// Checks if the set contains a specific range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range to check for containment
    ///
    /// # Returns
    ///
    /// `true` if the entire range is covered by ranges in the set, `false` otherwise.
    #[inline]
    #[must_use]
    pub fn contains(&self, rng: &Range) -> bool {
        if let Some((_, last)) = self.0.range(..=rng.start).next_back() {
            return rng.last <= *last;
        }
        false
    }

    /// Returns an iterator over the ranges in the set.
    ///
    /// The ranges are returned in ascending order by their start positions.
    #[inline]
    pub fn ranges(&self) -> impl Iterator<Item = Range> {
        self.0.iter().map(|(start, end)| Range::from((*start, *end)))
    }

    /// Inserts a range into the set.
    ///
    /// If the range overlaps or is adjacent to existing ranges, they will be merged.
    /// If the range is already fully contained in the set, nothing is changed.
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to insert
    ///
    /// # Returns
    ///
    /// `true` if the set was modified, `false` if the range was already fully contained.
    ///
    /// # Safety
    ///
    /// This method uses `unsafe` internally for performance optimization. The safety
    /// invariants are maintained by ensuring that the cursor operations are valid
    /// based on the preceding checks.
    pub fn insert_range(&mut self, rng: &Range) -> bool {
        // Position the cursor at the first position that might intersect with rng
        let mut cursor = self.0.upper_bound_mut(Bound::Included(&rng.start));
        if let Some(prev) = cursor.peek_prev().map(|(start, last)| Range::from((*start, *last)))
            && prev.intersects_or_adjacent(rng)
        {
            cursor.prev();
        }
        // Check if this is a no-op (containment)
        // We only need to check the element at the current cursor position.
        // If that element (the first one that might intersect) already contains
        // the new range, then the insertion is a no-op, so return false.
        if let Some(next) = cursor.peek_next().map(|(start, last)| Range::from((*start, *last)))
            && next.contains(rng)
        {
            return false;
        }
        // If it's not a no-op, perform the merge/insertion logic
        // Since we've excluded the fully contained case, any subsequent operations
        // will necessarily modify the set
        let mut merged_rng = *rng;
        // Continue looping as long as the next element exists and intersects with our range
        unsafe {
            while cursor
                .peek_next()
                .map(|(start, last)| Range::new(*start, *last))
                .is_some_and(|next| merged_rng.intersects_or_adjacent(&next))
            {
                // SAFETY: We've confirmed `peek_next()` returns `Some` in the loop condition,
                // so calling `remove_next()` will not panic. Using `unwrap_unchecked` is a
                // micro-optimization to avoid a redundant check.
                let rng_to_merge: Range = cursor.remove_next().unwrap_unchecked().into();
                // SAFETY: The loop condition `intersects_or_adjacent` guarantees that `merge`
                // will return `Some`, so this unwrap is safe.
                merged_rng = merged_rng.union(&rng_to_merge).unwrap_unchecked();
            }
            cursor.insert_after(merged_rng.start, merged_rng.last).unwrap_unchecked();
        };
        true
    }

    /// Inserts `n` consecutive elements starting at offset `at`.
    ///
    /// This is a convenience method that creates a range from `at` to `at + n - 1`
    /// and inserts it into the set. If the resulting range overlaps or is adjacent
    /// to existing ranges, they will be merged.
    ///
    /// If `n` is 0, this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of consecutive elements to insert
    /// * `at` - The starting offset where elements should be inserted
    ///
    /// # Returns
    ///
    /// Nothing is returned. If `n` is 0, this method does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{RangeSet, Range};
    /// let mut set = RangeSet::new();
    /// set.insert_n_at(5, 10);
    /// // This inserts elements 10, 11, 12, 13, 14
    ///
    /// assert!(set.contains(&Range::new(10, 14)));
    /// assert!(!set.contains(&Range::new(10, 15))); // 15 is not included
    /// ```
    /// Inserts a range of the specified length at the given position.
    ///
    /// This method creates a new range starting at `at` with length `n` and
    /// inserts it into the range set, merging with any overlapping or adjacent
    /// ranges as necessary.
    ///
    /// # Arguments
    ///
    /// * `n` - The length of the range to insert
    /// * `at` - The starting position of the range
    ///
    /// # Panics
    ///
    /// Panics if `at + n` would overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::RangeSet;
    /// let mut set = RangeSet::new();
    /// set.insert_n_at(10, 5); // Inserts range 5-14
    /// assert!(set.contains_n(10));
    /// ```
    pub fn insert_n_at(&mut self, n: usize, at: usize) {
        if n == 0 {
            return;
        }
        assert!(at.checked_add(n) < Some(usize::MAX));
        let rng = unsafe { Range::new_unchecked(at, at + n - 1) };
        self.insert_range(&rng);
    }

    /// Inserts a range of the specified length at the given position without bounds checking.
    ///
    /// This method creates a new range starting at `at` with length `n` and
    /// inserts it into the range set, merging with any overlapping or adjacent
    /// ranges as necessary.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `at + n` does not overflow `usize`.
    /// This function performs no bounds checking beyond a debug assertion.
    ///
    /// # Arguments
    ///
    /// * `n` - The length of the range to insert
    /// * `at` - The starting position of the range
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::RangeSet;
    /// let mut set = RangeSet::new();
    /// // SAFETY: 5 + 10 does not overflow usize
    /// unsafe { set.insert_n_at_unchecked(10, 5) }; // Inserts range 5-14
    /// assert!(set.contains_n(10));
    /// ```
    pub unsafe fn insert_n_at_unchecked(&mut self, n: usize, at: usize) {
        if n == 0 {
            return;
        }
        debug_assert!(at.checked_add(n) < Some(usize::MAX));
        let rng = unsafe { Range::new_unchecked(at, at + n - 1) };
        self.insert_range(&rng);
    }

    /// Computes the union of two sets by merging all ranges.
    ///
    /// This method merges ranges from both sets, creating a new set that contains
    /// all ranges from both input sets.
    ///
    /// # Arguments
    ///
    /// * `other` - The other set to union with
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the union of both sets.
    #[must_use]
    pub fn union_merge(&self, other: &Self) -> Self {
        let mut result = BTreeMap::new();
        let mut self_it = self.0.iter().peekable();
        let mut other_it = other.0.iter().peekable();

        // Store the current range being built that might still be expanded.
        let mut cur_merged: Option<Range> = None;

        // Use an infinite loop, and handle all cases including termination conditions inside.
        loop {
            // From the heads of both iterators, select the range with the smaller 'start' value
            // as the next one to process.
            // This match structure cleanly handles all cases and naturally includes the loop exit point.
            let next_rng_tuple = unsafe {
                match (self_it.peek(), other_it.peek()) {
                    // Both iterators have elements, select the one with the earlier start.
                    (Some((ls, _)), Some((rs, _))) => {
                        if ls <= rs {
                            self_it.next().unwrap_unchecked() // Safe: we know peek() returned Some
                        } else {
                            other_it.next().unwrap_unchecked()
                        }
                    }
                    // Only self_iter has elements, take it.
                    (Some(_), None) => self_it.next().unwrap_unchecked(),
                    // Only other_iter has elements, take it.
                    (None, Some(_)) => other_it.next().unwrap_unchecked(),
                    // Both iterators exhausted, merge process complete.
                    (None, None) => break,
                }
            };
            // Convert the tuple to our range type
            let next_rng = Range::new(*next_rng_tuple.0, *next_rng_tuple.1);
            match cur_merged.as_mut() {
                // This is the first range, or we just completed a merge gap.
                // Directly make next_range the new merge starting point.
                None => {
                    cur_merged = Some(next_rng);
                }
                // There's a range currently being merged.
                Some(merged) if merged.intersects_or_adjacent(&next_rng) => {
                    // The new range overlaps or is adjacent to the current merged range, expand `merged`'s bounds.
                    // Modify directly since `as_mut()` provides a mutable reference.
                    merged.last = merged.last.max(next_rng.last);
                }
                Some(merged) => {
                    // The new range has a gap from the current merged range.
                    // This means `merged` is complete, store it in the result set.
                    result.insert(merged.start, merged.last);
                    // Make `next_range` the new merge starting point.
                    *merged = next_rng;
                }
            }
        }
        // After the loop, the last `current_merged` hasn't been stored in the result set yet.
        if let Some(last_rng) = cur_merged {
            result.insert(last_rng.start, last_rng.last);
        }
        Self(result)
    }

    /// Computes the union of two sets.
    ///
    /// This method chooses the most efficient algorithm based on the sizes of the sets.
    /// If one set is much smaller than the other, it inserts ranges from the smaller
    /// set into the larger one. Otherwise, it uses the merge-based approach.
    ///
    /// # Arguments
    ///
    /// * `other` - The other set to union with
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the union of both sets.
    #[must_use]
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let self_rng_count = self.ranges_count();
        let other_rng_count = other.ranges_count();
        let insert_cost_estimate = other_rng_count * self_rng_count.ilog2() as usize;
        let merge_cost_estimate = self_rng_count + other_rng_count;
        if insert_cost_estimate < merge_cost_estimate && other_rng_count < self_rng_count {
            let mut result = self.clone();
            for (&start, &last) in &other.0 {
                result.insert_range(&Range::new(start, last));
            }
            result
        } else {
            self.union_merge(other)
        }
    }

    /// Performs union operation and assigns the result to self.
    ///
    /// # Arguments
    ///
    /// * `other` - The other set to union with
    #[inline]
    fn union_assign(&mut self, other: &Self) {
        if self.0.is_empty() {
            self.0 = other.0.clone();
        }
        if other.0.is_empty() {
            return;
        }
        let self_rng_count = self.ranges_count();
        let other_rng_count = other.ranges_count();
        let insert_cost_estimate = other_rng_count * self_rng_count.ilog2() as usize;
        let merge_cost_estimate = self_rng_count + other_rng_count;
        if insert_cost_estimate < merge_cost_estimate && other_rng_count < self_rng_count {
            for (&start, &last) in &other.0 {
                self.insert_range(&Range::new(start, last));
            }
        } else {
            *self = self.union_merge(other);
        }
    }

    /// Computes the difference of two sets.
    ///
    /// Returns a new set containing all elements in `self` that are not in `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - The set to subtract
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the difference.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return self.clone();
        }

        let mut result = Self::new();
        let mut a_it = self.0.iter();
        let mut b_it = other.0.iter().peekable();

        // Get the first range from A
        let mut cur_a = unsafe {
            let (&start, &last) = a_it.next().unwrap_unchecked();
            Range::new(start, last)
        };

        loop {
            // Look at B's next range
            if let Some(&(&b_start, &b_last)) = b_it.peek() {
                let b_range = Range::new(b_start, b_last);

                // If b_range is completely before current_a, skip this b_range
                if b_range.last() < cur_a.start() {
                    b_it.next(); // Consume b_range
                    continue;
                }

                // If b_range is completely after current_a, current_a won't be trimmed further
                // Complete processing of current_a, then try to get the next from A
                if b_range.start() > cur_a.last() {
                    result.insert_range(&cur_a);
                    if let Some((&s, &l)) = a_it.next() {
                        cur_a = Range::new(s, l);
                        continue;
                    }
                    break;
                }
                // If b_range leaves a part before it in current_a
                if b_range.start() > cur_a.start() {
                    let prefix = Range::new(cur_a.start(), b_range.start() - 1);
                    result.insert_range(&prefix);
                }
                // Update current_a's start, skipping the part covered by b_range
                // If b_range.last() overflows, it means current_a is completely covered
                if let Some(new_start) = b_range.last().checked_add(1) {
                    // If new_start is beyond current_a's range
                    if new_start > cur_a.last() {
                        // current_a is completely processed, get the next one
                        cur_a = match a_it.next() {
                            Some((&s, &l)) => Range::new(s, l),
                            None => break, // A exhausted, end
                        };
                    } else {
                        // current_a still has remainder, update start and continue processing
                        cur_a = Range::new(new_start, cur_a.last());
                    }
                } else {
                    // b_range.last() is usize::MAX, no remainder possible after current_a
                    cur_a = match a_it.next() {
                        Some((&s, &l)) => Range::new(s, l),
                        None => break, // A exhausted, end
                    };
                }
            } else {
                result.insert_range(&cur_a);
                for (&start, &last) in a_it {
                    result.insert_range(&Range::new(start, last));
                }
                break; // End main loop
            }
        }
        result
    }

    /// Performs difference operation and assigns the result to self.
    ///
    /// Subtracts `other` from `self` and stores the result in `self`.
    ///
    /// # Arguments
    ///
    /// * `other` - The set to subtract
    #[inline]
    pub fn difference_assign(&mut self, other: &Self) {
        if self.0.is_empty() || other.0.is_empty() {
            return;
        }
        *self = self.difference(other);
    }

    /// Computes the union of a `RangeSet` with a `FrozenRangeSet`.
    ///
    /// Creates a new `RangeSet` that is the union of `self` and a `FrozenRangeSet`.
    ///
    /// This method returns a new range set containing all elements from both sets,
    /// merging overlapping and adjacent ranges as necessary.
    ///
    /// # Arguments
    ///
    /// * `other` - The frozen range set to union with
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the union of both sets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{RangeSet, Range};
    /// let mut set1 = RangeSet::new();
    /// set1.insert_range(&Range::new(0, 10));
    /// let mut set2 = RangeSet::new();
    /// set2.insert_range(&Range::new(5, 15));
    /// let frozen = set2.freeze();
    /// let result = set1.union_frozen(&frozen);
    /// assert_eq!(result.len(), 16);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `other` - The frozen range set to union with
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the union of both sets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{RangeSet, Range};
    /// let mut set1 = RangeSet::new();
    /// set1.insert_range(&Range::new(0, 10));
    /// let mut set2 = RangeSet::new();
    /// set2.insert_range(&Range::new(5, 15));
    /// let frozen = set2.freeze();
    /// let result = set1.union_frozen(&frozen);
    /// assert_eq!(result.len(), 16);
    /// ```
    #[must_use]
    #[inline]
    pub fn union_frozen(&self, other: &FrozenRangeSet) -> Self {
        if self.0.is_empty() {
            return other.clone().into();
        }
        if other.is_empty() {
            return self.clone();
        }
        let mut result = self.clone();
        for range in other.iter() {
            result.insert_range(range);
        }
        result
    }

    /// Performs union operation between `RangeSet` and `FrozenRangeSet` and assigns the result to self.
    ///
    /// # Arguments
    ///
    /// * `other` - The frozen range set to union with
    #[inline]
    pub fn union_assign_frozen(&mut self, other: &FrozenRangeSet) {
        if self.0.is_empty() {
            *self = other.clone().into();
            return;
        }
        if other.is_empty() {
            return;
        }
        for range in other.iter() {
            self.insert_range(range);
        }
    }

    /// Computes the difference of a `RangeSet` with a `FrozenRangeSet`.
    ///
    /// Returns a new set containing all elements in `self` that are not in `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - The frozen range set to subtract
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the difference.
    #[must_use]
    #[inline]
    pub fn difference_frozen(&self, other: &FrozenRangeSet) -> Self {
        if self.0.is_empty() || other.is_empty() {
            return self.clone();
        }

        let other_set: Self = other.clone().into();
        self.difference(&other_set)
    }

    /// Performs difference operation between `RangeSet` and `FrozenRangeSet` and assigns the result to self.
    ///
    /// Subtracts `other` from `self` and stores the result in `self`.
    ///
    /// # Arguments
    ///
    /// * `other` - The frozen range set to subtract
    #[inline]
    pub fn difference_assign_frozen(&mut self, other: &FrozenRangeSet) {
        if self.0.is_empty() || other.is_empty() {
            return;
        }
        let other_set: Self = other.clone().into();
        self.difference_assign(&other_set);
    }

    /// Creates a frozen version of the range set.
    ///
    /// A frozen range set is an immutable snapshot of the current range set
    /// that can be shared across threads or stored for later use.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 5));
    /// set.insert_range(&Range::new(10, 15));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen.len(), 2); // 2 ranges
    /// ```
    #[must_use]
    /// Creates a frozen version of this range set.
    ///
    /// Returns a `FrozenRangeSet` that contains all the ranges from this set.
    /// Frozen range sets are optimized for read-only operations and have
    /// different performance characteristics compared to mutable range sets.
    ///
    /// # Returns
    ///
    /// A `FrozenRangeSet` containing all ranges from this set.
    pub fn freeze(&self) -> FrozenRangeSet {
        let ranges = self.0.iter().map(|(&start, &last)| Range::new(start, last)).collect::<Box<[_]>>();
        FrozenRangeSet(ranges)
    }

    /// Creates a chunking iterator over the set.
    ///
    /// This method creates an iterator that consumes the set and produces
    /// chunks of ranges with a specified maximum size.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The target size for each chunk
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is 0 (in debug builds only).
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 100));
    /// let chunks: Vec<_> = set.into_chunks(10).collect();
    /// assert_eq!(chunks.len(), 11); // 101 elements in chunks of 10
    /// ```
    /// Creates an iterator that yields chunks of the range set.
    ///
    /// This method consumes ranges from the set and yields chunks where each chunk
    /// contains approximately `block_size` elements. Each chunk is returned as a
    /// `FrozenRangeSet`.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The size of each chunk
    ///
    /// # Returns
    ///
    /// A `ChunkedMutIter` that yields mutable chunks of the range set.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is 0 (in debug builds).
    pub fn into_chunks(&mut self, block_size: usize) -> ChunkedMutIter<'_> {
        debug_assert!(block_size > 0, "block_size must be greater than 0");
        ChunkedMutIter { inner: self, block_size }
    }
}

impl BitOrAssign<&Self> for RangeSet {
    /// Performs the `|=` operation, equivalent to [`RangeSet::union_assign`](RangeSet::union).
    #[inline]
    fn bitor_assign(&mut self, rhs: &Self) { self.union_assign(rhs); }
}

impl BitOr<Self> for &RangeSet {
    type Output = RangeSet;

    /// Performs the `|` operation, equivalent to [`RangeSet::union`].
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output { self.union(rhs) }
}

impl SubAssign<&Self> for RangeSet {
    /// Performs the `-=` operation, equivalent to [`RangeSet::difference_assign`](RangeSet::difference_assign).
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) { self.difference_assign(rhs); }
}

impl Sub<Self> for &RangeSet {
    type Output = RangeSet;

    /// Performs the `-` operation, equivalent to [`RangeSet::difference`].
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output { self.difference(rhs) }
}

/// An iterator that chunks an `RangeSet` into fixed-size blocks.
///
/// This iterator consumes an `RangeSet` and produces chunks of ranges
/// where each chunk has approximately the specified block size.
/// An iterator that yields chunks of a `RangeSet`.
///
/// This iterator consumes the underlying `RangeSet` and produces chunks of ranges
/// with a total size approximately equal to the configured block size. Each chunk
/// is returned as a `FrozenRangeSet` containing the ranges that fit within the block.
///
/// # Examples
///
/// ```
/// # use sparse_ranges::{Range, RangeSet};
/// let mut set = RangeSet::new();
/// set.insert_range(&Range::new(0, 100));
/// let chunks: Vec<_> = set.into_chunks(10).collect();
/// assert_eq!(chunks.len(), 11); // 101 elements in chunks of 10
/// ```
pub struct ChunkedMutIter<'a> {
    inner: &'a mut RangeSet,
    block_size: usize,
}

impl Iterator for ChunkedMutIter<'_> {
    type Item = FrozenRangeSet;

    /// Produces the next chunk of ranges.
    ///
    /// This method consumes ranges from the underlying set and produces
    /// a boxed slice of ranges with a total size approximately equal
    /// to the configured block size.
    ///
    /// # Returns
    ///
    /// A boxed slice of ranges representing the next chunk, or `None`
    /// if the set has been fully consumed.
    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.is_empty() {
            return None;
        }
        let mut chunk_rngs = Vec::with_capacity(1);
        let mut remaining_size = self.block_size;
        while remaining_size > 0 {
            // First, peek at what the first range is, but don't remove it yet
            let Some((&start, &last)) = self.inner.0.first_key_value() else {
                // If the BTreeMap becomes empty during the loop, break out
                break;
            };
            let cur_rng_len = last - start + 1;
            if cur_rng_len <= remaining_size {
                // --- The current range can be fully included in the chunk ---
                // Remove this range from the BTreeMap
                self.inner.0.pop_first();
                // Add it to the current chunk
                chunk_rngs.push(Range::new(start, last));
                // Update the chunk's remaining capacity
                remaining_size -= cur_rng_len;
            } else {
                // --- The current range is too large, only part of it fits ---
                // Calculate the end position that this chunk can accommodate from this range
                debug_assert!(start.checked_add(remaining_size) < Some(usize::MAX));
                let chunk_last = start + remaining_size - 1;
                // Add that part to the chunk
                chunk_rngs.push(Range::new(start, chunk_last));
                // So we remove the old entry, then insert a new entry representing the remainder.
                let original_last = self.inner.0.pop_first().unwrap().1;
                debug_assert!(chunk_last < usize::MAX);
                self.inner.0.insert(chunk_last + 1, original_last);
                // The chunk is now full, force the loop to end
                remaining_size = 0;
            }
        }
        // If we successfully got any data from the BTreeMap,
        // return the constructed chunk, otherwise return None.
        chunk_rngs.is_empty().not().then(|| FrozenRangeSet(chunk_rngs.into_boxed_slice()))
    }
}

impl<T: Into<Range>> FromIterator<T> for RangeSet {
    /// Creates an `RangeSet` from an iterator.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator of items that can be converted to `Range`
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing all the ranges from the iterator.
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        for item in iter {
            set.insert_range(&item.into());
        }
        set
    }
}

/// Error types that can occur when working with range sets.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// An error occurred when parsing a range header.
    #[cfg(feature = "http")]
    #[error(transparent)]
    Header(#[from] http_range_header::RangeUnsatisfiableError),
    /// The range unit is invalid or unsupported.
    #[error("invalid range unit")]
    Invalid,
    /// An index overflow or underflow occurred during range computation.
    #[error("index overflow")]
    IndexOverflow,
    /// The resulting range set is empty.
    #[error("empty ranges")]
    Empty,
}

#[cfg(feature = "http")]
impl RangeSet {
    /// Parses an HTTP 'Range' header string relative to a total entity size.
    ///
    /// This function correctly handles all valid range formats as per RFC 7233, including:
    /// - `bytes=0-499` (absolute range)
    /// - `bytes=500-` (open-ended range from 500 to end)
    /// - `bytes=-100` (suffix range, last 100 bytes)
    ///
    /// Ranges are automatically merged if they overlap or are adjacent.
    ///
    /// # Errors
    ///
    /// This function will return an error in the following situations:
    ///
    /// * [`Error::Header`] - If the header string cannot be parsed according to HTTP range header format
    /// * [`Error::Invalid`] - If any of the ranges are invalid (e.g., start position greater than end position, or
    ///   start position is greater than or equal to the total size)
    /// * [`Error::Empty`] - If the total size is 0, or if all ranges are unsatisfiable resulting in an empty set
    ///
    /// Parses HTTP Range headers into a `RangeSet`.
    ///
    /// This method parses an HTTP Range header (as defined in RFC 7233) and
    /// converts it to a `RangeSet`. It handles various range formats including
    /// byte ranges, suffix ranges, and multi-range requests.
    ///
    /// # Arguments
    ///
    /// * `header_content` - The raw HTTP Range header value (without "Range:" prefix)
    /// * `total_size` - The total size of the resource being requested
    ///
    /// # Returns
    ///
    /// A `RangeSet` containing the parsed ranges if successful, or an `Error`
    /// if the header is invalid or unsatisfiable.
    ///
    /// # Errors
    ///
    /// Returns `Error::Empty` if the range is empty or if `total_size` is 0.
    /// Returns `Error::Invalid` if the range format is invalid or unsatisfiable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::RangeSet;
    /// let set = RangeSet::parse_ranges_headers("bytes=0-499", 1000).unwrap();
    /// assert_eq!(set.len(), 500);
    /// ```
    pub fn parse_ranges_headers(header_content: &str, total_size: usize) -> Result<Self, Error> {
        use http_range_header::{EndPosition, StartPosition};
        if total_size == 0 {
            // According to RFC 7233, a range header on a zero-length entity
            // is always unsatisfiable.
            return Err(Error::Empty);
        }

        let mut set = Self::new();
        // The library already handles parsing the string format.
        let rngs = http_range_header::parse_range_header(header_content)?.ranges;

        for item in rngs {
            let (start, last) = match (item.start, item.end) {
                // `bytes=A-B` (e.g., `bytes=0-499`)
                (StartPosition::Index(s), EndPosition::Index(l)) => (s as usize, l as usize),
                // `bytes=A-` (e.g., `bytes=500-`)
                (StartPosition::Index(s), EndPosition::LastByte) => {
                    let start = s as usize;
                    if start >= total_size {
                        return Err(Error::Invalid);
                    }
                    (start, total_size - 1)
                }
                // `bytes=-C` (e.g., `bytes=-100`)
                (StartPosition::FromLast(c), EndPosition::LastByte) => {
                    // `-0`: Suffix length cannot be 0.
                    if c == 0 {
                        return Err(Error::Empty);
                    }
                    // Calculate start, avoiding underflow if c > total_size.
                    let s = total_size.saturating_sub(c as usize);
                    (s, total_size - 1)
                }
                (StartPosition::FromLast(_), EndPosition::Index(_)) => return Err(Error::Invalid),
            };

            // --- Validation as per RFC 7233 ---
            // "If the last-byte-pos value is present, it MUST be greater than or
            // equal to the first-byte-pos in that byte-range-spec."
            if start > last {
                return Err(Error::Invalid);
            }

            // "if the first-byte-pos of all of the byte-range-spec values
            // is greater than or equal to the current length of the representation,
            // the server SHOULD send a 416 (Range Not Satisfiable) response."
            // We check this for each range.
            if start >= total_size {
                return Err(Error::Invalid);
            }

            // The range is valid, but we need to clamp `last` to the actual size.
            // For example, a request for `0-1000` on a 500-byte file should yield `0-499`.
            let last_clamped: usize = last.min(total_size - 1);
            set.insert_range(&Range::new(start, last_clamped));
        }
        // If after all processing the set is empty (e.g., all ranges were invalid
        // in a way that didn't trigger an early return, although unlikely with current logic),
        // it might also be considered unsatisfiable.
        if set.is_empty() {
            return Err(Error::Empty);
        }
        Ok(set)
    }

    /// Converts the set to an HTTP range header string.
    ///
    /// # Returns
    ///
    /// A boxed string representing the ranges in HTTP header format (e.g., "bytes=0-100,200-300"),
    /// or `None` if the set is empty.
    #[inline]
    #[must_use]
    /// Converts the range set to an HTTP Range header string format.
    ///
    /// Returns a string in the format "bytes=start1-end1,start2-end2,..." suitable
    /// for use in HTTP Range headers.
    ///
    /// # Returns
    ///
    /// An `Option` containing the formatted string if the range set is not empty,
    /// or `None` if the range set is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{RangeSet, Range};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 499));
    /// set.insert_range(&Range::new(1000, 1499));
    /// assert_eq!(set.to_http_range_header().unwrap().as_ref(), "bytes=0-499,1000-1499");
    /// ```
    pub fn to_http_range_header(&self) -> Option<Box<str>> {
        if self.0.is_empty() {
            return None;
        }
        let parts: Box<[String]> = self.0.iter().map(|(&start, &last)| format!("{start}-{last}")).collect();
        Some(format!("bytes={}", parts.join(",")).into_boxed_str())
    }
}

/// An immutable collection of non-overlapping, sorted ranges.
///
/// A `FrozenRangeSet` is a snapshot of a `RangeSet` that cannot be modified.
/// It is optimized for read-only operations and can be shared across threads
/// or stored for later use without the overhead of mutable operations.
///
/// Frozen range sets are particularly useful when you need to:
/// - Store a range set that won't change
/// - Share ranges between multiple threads without locks
/// - Cache intermediate results of range operations
#[derive(Clone, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrozenRangeSet(Box<[Range]>);

impl Deref for FrozenRangeSet {
    type Target = [Range];

    /// Dereferences to a slice of `Range`s.
    ///
    /// This allows treating a `FrozenRangeSet` as a slice, enabling
    /// convenient access via indexing and iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// set.insert_range(&Range::new(20, 30));
    /// let frozen = set.freeze();
    ///
    /// // Access ranges by index
    /// assert_eq!(frozen[0], Range::new(0, 10));
    ///
    /// // Iterate over ranges
    /// for range in &*frozen {
    ///     println!("{:?}", range);
    /// }
    /// ```
    #[inline]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl FrozenRangeSet {
    /// Returns the start offset of the first range in the set.
    ///
    /// # Returns
    ///
    /// `None` if the set is empty, `Some(start)` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(10, 20));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen.start(), Some(10));
    /// ```
    #[inline]
    #[must_use]
    pub fn start(&self) -> Option<usize> { self.0.first().map(Range::start) }

    /// Returns the end offset of the last range in the set.
    ///
    /// # Returns
    ///
    /// `None` if the set is empty, `Some(last)` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(10, 20));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen.last(), Some(20));
    /// ```
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<usize> { self.0.last().map(Range::last) }

    /// Returns the number of separate ranges in the frozen set.
    ///
    /// Note: This returns the count of ranges, not the total number of elements.
    /// Use `iter().map(|r| r.len()).sum()` to get the total element count.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 5));
    /// set.insert_range(&Range::new(10, 15));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen.ranges_count(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn ranges_count(&self) -> usize { self.0.len() }

    /// Checks if the set contains a specific offset.
    ///
    /// # Arguments
    ///
    /// * `n` - The offset to check
    ///
    /// # Returns
    ///
    /// `true` if `n` is within any range in the set, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 5));
    /// set.insert_range(&Range::new(10, 15));
    /// let frozen = set.freeze();
    /// assert!(frozen.contains_n(3));
    /// assert!(!frozen.contains_n(7));
    /// assert!(frozen.contains_n(12));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_n(&self, n: usize) -> bool {
        let partition_idx = self.0.partition_point(|rng| rng.start() <= n);
        if partition_idx == 0 {
            return false;
        }
        let candidate_rng = unsafe { self.0.get_unchecked(partition_idx - 1) };
        n <= candidate_rng.last()
    }

    /// Checks if the set contains a specific range.
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to check for containment
    ///
    /// # Returns
    ///
    /// `true` if the entire range is covered by ranges in the set, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// let frozen = set.freeze();
    /// assert!(frozen.contains(&Range::new(2, 8)));
    /// assert!(!frozen.contains(&Range::new(5, 15)));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, rng: &Range) -> bool {
        let partition_idx = self.0.partition_point(|r| r.start() <= rng.start());
        if partition_idx == 0 {
            return false;
        }
        let candidate_rng = unsafe { self.0.get_unchecked(partition_idx - 1) };
        candidate_rng.contains(rng)
    }

    /// Returns `true` if the set contains no ranges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::RangeSet;
    /// let set = RangeSet::new();
    /// let frozen = set.freeze();
    /// assert!(frozen.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool { self.0.is_empty() }
}

impl BitOr<&FrozenRangeSet> for &RangeSet {
    type Output = RangeSet;

    /// Performs the `|` operation between `RangeSet` and `FrozenRangeSet`.
    #[inline]
    fn bitor(self, rhs: &FrozenRangeSet) -> Self::Output { self.union_frozen(rhs) }
}

impl Sub<&FrozenRangeSet> for &RangeSet {
    type Output = RangeSet;

    /// Performs the `-` operation between `RangeSet` and `FrozenRangeSet`.
    #[inline]
    fn sub(self, rhs: &FrozenRangeSet) -> Self::Output { self.difference_frozen(rhs) }
}

impl BitOrAssign<&FrozenRangeSet> for RangeSet {
    /// Performs the `|=` operation between `RangeSet` and `FrozenRangeSet`.
    #[inline]
    fn bitor_assign(&mut self, rhs: &FrozenRangeSet) { self.union_assign_frozen(rhs); }
}

impl SubAssign<&FrozenRangeSet> for RangeSet {
    /// Performs the `-=` operation between `RangeSet` and `FrozenRangeSet`.
    #[inline]
    fn sub_assign(&mut self, rhs: &FrozenRangeSet) { self.difference_assign_frozen(rhs); }
}

#[cfg(feature = "http")]
impl FrozenRangeSet {
    #[inline]
    #[must_use]
    /// Converts the frozen range set to an HTTP Range header string format.
    ///
    /// Returns a string in the format "bytes=start1-end1,start2-end2,..." suitable
    /// for use in HTTP Range headers.
    ///
    /// # Returns
    ///
    /// An `Option` containing the formatted string if the range set is not empty,
    /// or `None` if the range set is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{RangeSet, Range};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 499));
    /// set.insert_range(&Range::new(1000, 1499));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen.to_http_range_header().unwrap().as_ref(), "bytes=0-499,1000-1499");
    /// ```
    pub fn to_http_range_header(&self) -> Option<Box<str>> {
        if self.is_empty() {
            return None;
        }
        let parts: Box<[String]> = self.iter().map(|range| format!("{}-{}", range.start(), range.last())).collect();
        Some(format!("bytes={}", parts.join(",")).into_boxed_str())
    }
}

impl From<RangeSet> for FrozenRangeSet {
    /// Converts a mutable `RangeSet` into an immutable `FrozenRangeSet`.
    ///
    /// # Arguments
    ///
    /// * `set` - The range set to freeze
    ///
    /// # Returns
    ///
    /// A new `FrozenRangeSet` containing the same ranges as the input set.
    #[inline]
    fn from(set: RangeSet) -> Self {
        let ranges = set.0.into_iter().map(Into::into).collect::<Box<[_]>>();
        Self(ranges)
    }
}

impl From<FrozenRangeSet> for RangeSet {
    /// Converts a `FrozenRangeSet` back into a mutable `RangeSet`.
    ///
    /// # Arguments
    ///
    /// * `frozen` - The frozen range set to convert
    ///
    /// # Returns
    ///
    /// A new `RangeSet` containing the same ranges as the frozen set.
    #[inline]
    fn from(frozen: FrozenRangeSet) -> Self {
        let map = frozen.0.into_iter().map(|Range { start, last }| (start, last)).collect::<BTreeMap<_, _>>();
        Self(map)
    }
}

impl PartialEq<FrozenRangeSet> for RangeSet {
    /// Checks if a `RangeSet` is equal to a `FrozenRangeSet`.
    ///
    /// This method returns `true` if both sets contain the same ranges in the same order,
    /// regardless of their mutability status.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// set.insert_range(&Range::new(20, 30));
    /// let frozen = set.freeze();
    /// assert_eq!(set, frozen);
    ///
    /// let mut set2 = RangeSet::new();
    /// set2.insert_range(&Range::new(0, 10));
    /// assert_ne!(set2, frozen);
    /// ```
    #[inline]
    fn eq(&self, other: &FrozenRangeSet) -> bool {
        self.ranges_count() == other.ranges_count() && self.ranges().eq(other.iter().copied())
    }
}

impl PartialEq<RangeSet> for FrozenRangeSet {
    /// Checks if a `FrozenRangeSet` is equal to a `RangeSet`.
    ///
    /// This method delegates to the `PartialEq<FrozenRangeSet>` implementation for `RangeSet`,
    /// so it returns `true` if both sets contain the same ranges in the same order,
    /// regardless of their mutability status.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// set.insert_range(&Range::new(20, 30));
    /// let frozen = set.freeze();
    /// assert_eq!(frozen, set);
    /// ```
    #[inline]
    fn eq(&self, other: &RangeSet) -> bool { other.eq(self) }
}

impl PartialEq<Range> for FrozenRangeSet {
    /// Checks if a `FrozenRangeSet` is equal to a single `Range`.
    ///
    /// This method returns `true` if the `FrozenRangeSet` contains exactly one range
    /// and that range is equal to the provided `Range`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let range = Range::new(0, 10);
    /// let mut set = RangeSet::new();
    /// set.insert_range(&range);
    /// let frozen = set.freeze();
    /// assert_eq!(frozen, range);
    ///
    /// let mut set2 = RangeSet::new();
    /// set2.insert_range(&Range::new(0, 5));
    /// set2.insert_range(&Range::new(7, 10)); // Note: gap between 5 and 7
    /// let frozen2 = set2.freeze();
    /// assert_ne!(frozen2, range); // frozen2 has two ranges
    /// ```
    #[inline]
    fn eq(&self, other: &Range) -> bool { self.0.len() == 1 && *unsafe { self.0.first().unwrap_unchecked() } == *other }
}

impl PartialEq<FrozenRangeSet> for Range {
    /// Checks if a `Range` is equal to a `FrozenRangeSet`.
    ///
    /// This method delegates to the `PartialEq<Range>` implementation for `FrozenRangeSet`,
    /// so it returns `true` if the `FrozenRangeSet` contains exactly one range and that range
    /// is equal to this `Range`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let range = Range::new(0, 10);
    /// let mut set = RangeSet::new();
    /// set.insert_range(&range);
    /// let frozen = set.freeze();
    /// assert_eq!(range, frozen);
    /// ```
    #[inline]
    fn eq(&self, other: &FrozenRangeSet) -> bool { other.eq(self) }
}

/// The base value for binary units (1024).
const BINARY_BASE: usize = 1024;

/// Table of binary units and their corresponding names.
///
/// Each tuple contains a multiplier and the unit name for binary data sizes,
/// from bytes (B) up to exbibytes (EiB).
const BINARY_UNIT_TABLE: [(usize, &str); 7] = [
    (1, "B"),
    (BINARY_BASE, "KiB"),
    (BINARY_BASE.pow(2), "MiB"),
    (BINARY_BASE.pow(3), "GiB"),
    (BINARY_BASE.pow(4), "TiB"),
    (BINARY_BASE.pow(5), "PiB"),
    (BINARY_BASE.pow(6), "EiB"),
];

/// The base value for SI (decimal) units (1000).
const SI_BASE: usize = 1000;

/// Table of SI (decimal) units and their corresponding names.
///
/// Each tuple contains a multiplier and the unit name for decimal data sizes,
/// from bytes (B) up to exabytes (EB).
const SI_UNIT_TABLE: [(usize, &str); 7] = [
    (1, "B"),
    (SI_BASE, "KB"),
    (SI_BASE.pow(2), "MB"),
    (SI_BASE.pow(3), "GB"),
    (SI_BASE.pow(4), "TB"),
    (SI_BASE.pow(5), "PB"),
    (SI_BASE.pow(6), "EB"),
];

/// Analyzes a byte size and returns an appropriate value, unit name, and base.
///
/// This function converts a byte size to a human-readable format by selecting
/// the appropriate unit (B, KB, MB, etc.) and calculating the value in that unit.
/// This is used internally for Display formatting of ranges.
///
/// # Arguments
///
/// * `size` - The size in bytes to analyze
/// * `use_binary` - If true, use binary units (KiB, MiB, etc. with base 1024), if false, use decimal units (KB, MB,
///   etc. with base 1000)
///
/// # Returns
///
/// A tuple containing:
/// * The value expressed in the selected unit
/// * The unit name as a string slice
/// * The base multiplier used for the unit
fn analyze_bytes(size: usize, use_binary: bool) -> (f64, &'static str, usize) {
    if size == 0 {
        return (0., "B", 1);
    }
    let (base, unit_table) = if use_binary {
        (BINARY_BASE as f64, &BINARY_UNIT_TABLE)
    } else {
        (SI_BASE as f64, &SI_UNIT_TABLE)
    };
    let exp = if size > 0 {
        (size as f64).log(base).floor() as usize
    } else {
        0
    };
    let idx = exp.min(unit_table.len() - 1);
    let (unit_base, unit_name) = unit_table[idx];
    let val = size as f64 / unit_base as f64;
    (val, unit_name, unit_base)
}

impl Display for Range {
    /// Formats the `Range` for display purposes.
    ///
    /// This implementation displays the range in a human-readable format with
    /// size units. Use the default format (`{}`) for binary units (KiB, MiB, etc.)
    /// or alternate format (`{:#}`) for SI units (KB, MB, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::Range;
    /// let range = Range::new(0, 1023);
    /// println!("{}", range); // "0..=1023 (1.00 KiB)"
    ///
    /// let range2 = Range::new(0, 1000);
    /// println!("{:#}", range2); // "0..=1000 (1000 B)" with SI units
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let use_binary = !f.alternate();
        let (_last_val_temp, common_unit, unit_base) = analyze_bytes(self.last, use_binary);
        let unit_base_f64 = unit_base as f64;
        let format_num = |val: f64| -> String {
            if (val.fract()).abs() < 1e-9 {
                format!("{val:.0}")
            } else {
                format!("{val:.2}")
            }
        };
        let start_val = self.start as f64 / unit_base_f64;
        let last_val = self.last as f64 / unit_base_f64;
        if self.start == self.last {
            write!(f, "{} {}", format_num(start_val), common_unit)
        } else {
            write!(f, "{} ~ {} {}", format_num(start_val), format_num(last_val), common_unit)
        }
    }
}

impl Display for RangeSet {
    /// Formats the `RangeSet` for display purposes.
    ///
    /// This implementation displays the range set in a human-readable format with
    /// size units, showing all ranges separated by commas. Use the default format (`{}`)
    /// for binary units (KiB, MiB, etc.) or alternate format (`{:#}`) for SI units (KB, MB, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 1023));
    /// set.insert_range(&Range::new(2048, 3071));
    /// println!("{}", set); // "0 ~ 1 KiB, 2 ~ 3 KiB"
    ///
    /// println!("{:#}", set); // with SI units
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (&start, &last) in &self.0 {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{}", Range::new(start, last))?;
            first = false;
        }
        Ok(())
    }
}

impl Debug for FrozenRangeSet {
    /// Formats the `FrozenRangeSet` for debugging purposes.
    ///
    /// This implementation displays the frozen range set in a human-readable format,
    /// showing all the ranges contained in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 10));
    /// set.insert_range(&Range::new(20, 30));
    /// let frozen = set.freeze();
    /// assert_eq!(format!("{:?}", frozen), "FrozenRangeSet {0..=10, 20..=30}");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrozenRangeSet ")?;
        let mut set = f.debug_set();
        for rng in &self.0 {
            set.entry(&format_args!("{}..={}", rng.start(), rng.last()));
        }
        set.finish()
    }
}

impl Display for FrozenRangeSet {
    /// Formats the `FrozenRangeSet` for display purposes.
    ///
    /// This implementation displays the frozen range set in a human-readable format with
    /// size units, showing all ranges separated by commas. Use the default format (`{}`)
    /// for binary units (KiB, MiB, etc.) or alternate format (`{:#}`) for SI units (KB, MB, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{Range, RangeSet};
    /// let mut set = RangeSet::new();
    /// set.insert_range(&Range::new(0, 1023));
    /// set.insert_range(&Range::new(2048, 3071));
    /// let frozen = set.freeze();
    /// println!("{}", frozen); // "0 ~ 1 KiB, 2 ~ 3 KiB"
    ///
    /// println!("{:#}", frozen); // with SI units
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for rng in &self.0 {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{rng}")?;
            first = false;
        }
        Ok(())
    }
}
#[cfg(test)]
#[allow(clippy::pedantic)]
mod tests {
    #[cfg(feature = "http")]
    use crate::Error;
    use crate::{FrozenRangeSet, Range, RangeSet};

    // --- Helper Functions ---
    fn make_set(ranges: &[(usize, usize)]) -> RangeSet {
        let mut set = RangeSet::new();
        for &(start, last) in ranges {
            set.insert_range(&Range::new(start, last));
        }
        set
    }

    fn check_ranges(set: &RangeSet, expected: &[(usize, usize)]) {
        let ranges: Vec<(usize, usize)> = set.ranges().map(|r| (r.start(), r.last())).collect();
        assert_eq!(ranges, expected, "RangeSet ranges do not match expected");
    }

    // --- Range Tests ---

    #[test]
    fn test_range_new_valid() {
        let r = Range::new(10, 20);
        assert_eq!(r.start(), 10);
        assert_eq!(r.last(), 20);
        assert_eq!(r.len(), 11);
        assert!(!r.is_empty());
    }

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_range_new_panic_order() { let _ = Range::new(20, 10); }

    #[test]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_panic_max() { let _ = Range::new(0, usize::MAX); }

    #[test]
    fn test_range_edge_cases() {
        // Single point
        let r = Range::new(5, 5);
        assert_eq!(r.len(), 1);

        // Max valid range
        let r = Range::new(0, usize::MAX - 1);
        assert_eq!(r.len(), usize::MAX); // Logic check: (MAX-1) - 0 + 1 = MAX

        // High values
        let r = Range::new(usize::MAX - 2, usize::MAX - 1);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_range_contains_n() {
        let r = Range::new(10, 20);
        assert!(r.contains_n(10));
        assert!(r.contains_n(15));
        assert!(r.contains_n(20));
        assert!(!r.contains_n(9));
        assert!(!r.contains_n(21));
    }

    #[test]
    fn test_range_contains_range() {
        let r = Range::new(10, 30);
        assert!(r.contains(&Range::new(10, 30))); // Self
        assert!(r.contains(&Range::new(15, 25))); // Inner
        assert!(r.contains(&Range::new(10, 15))); // Touch start
        assert!(r.contains(&Range::new(25, 30))); // Touch end

        assert!(!r.contains(&Range::new(9, 30))); // Extend left
        assert!(!r.contains(&Range::new(10, 31))); // Extend right
        assert!(!r.contains(&Range::new(5, 40))); // Superset
        assert!(!r.contains(&Range::new(40, 50))); // Disjoint
    }

    #[test]
    fn test_range_intersects() {
        let r = Range::new(10, 20);

        // Overlap
        assert!(r.intersects(&Range::new(5, 15)));
        assert!(r.intersects(&Range::new(15, 25)));
        assert!(r.intersects(&Range::new(12, 18)));
        assert!(r.intersects(&Range::new(5, 25)));

        // Touch (is intersection)
        assert!(r.intersects(&Range::new(0, 10)));
        assert!(r.intersects(&Range::new(20, 30)));

        // Disjoint
        assert!(!r.intersects(&Range::new(0, 9)));
        assert!(!r.intersects(&Range::new(21, 30)));
    }

    #[test]
    fn test_range_adjacency() {
        let r = Range::new(10, 20);

        // Exactly adjacent
        assert!(r.is_adjacent(&Range::new(5, 9)));
        assert!(r.is_adjacent(&Range::new(21, 25)));

        // Overlapping is NOT adjacent
        assert!(!r.is_adjacent(&Range::new(5, 10)));

        // Gap > 0 is NOT adjacent
        assert!(!r.is_adjacent(&Range::new(0, 8)));
        assert!(!r.is_adjacent(&Range::new(22, 30)));
    }

    #[test]
    fn test_range_operations() {
        let r = Range::new(10, 20);

        // Midpoint
        assert_eq!(r.midpoint(), 15);
        assert_eq!(Range::new(10, 11).midpoint(), 10);

        // Intersection
        assert_eq!(r.intersection(&Range::new(15, 25)), Some(Range::new(15, 20)));
        assert_eq!(r.intersection(&Range::new(21, 30)), None);

        // Union (Merge)
        assert_eq!(r.union(&Range::new(15, 25)), Some(Range::new(10, 25))); // Overlap
        assert_eq!(r.union(&Range::new(21, 25)), Some(Range::new(10, 25))); // Adjacent
        assert_eq!(r.union(&Range::new(22, 25)), None); // Gap

        // Difference
        // 1. Remove middle (hole) - not supported by single Range return type logic, but `difference` returns
        //    (Option<Range>, Option<Range>)
        assert_eq!(r.difference(&Range::new(12, 18)), (Some(Range::new(10, 11)), Some(Range::new(19, 20))));
        // 2. Remove start
        assert_eq!(r.difference(&Range::new(5, 15)), (None, Some(Range::new(16, 20))));
        // 3. Remove end
        assert_eq!(r.difference(&Range::new(15, 25)), (Some(Range::new(10, 14)), None));
        // 4. Remove all
        assert_eq!(r.difference(&Range::new(0, 50)), (None, None));
        // 5. No overlap
        assert_eq!(r.difference(&Range::new(30, 40)), (Some(r), None));
    }

    // --- Range Conversion Tests ---

    #[test]
    fn test_range_conversions() {
        // TryFrom<&ops::Range<usize>> - success
        let std_range = 10..20;
        let range = Range::try_from(&std_range).unwrap();
        assert_eq!(range.start(), 10);
        assert_eq!(range.last(), 19); // exclusive end becomes inclusive

        // TryFrom<&ops::Range<usize>> - error (empty range causes underflow)
        let empty_range = 0..0;
        assert!(Range::try_from(&empty_range).is_err());

        // From<&ops::RangeInclusive<usize>>
        let inclusive_range = 10..=20;
        let range = Range::from(&inclusive_range);
        assert_eq!(range.start(), 10);
        assert_eq!(range.last(), 20);

        // From<(usize, usize)>
        let range = Range::from((10, 20));
        assert_eq!(range.start(), 10);
        assert_eq!(range.last(), 20);
    }

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_range_from_tuple_panic() { let _ = Range::from((20, 10)); }

    #[test]
    fn test_range_new_unchecked() {
        // Test unsafe new_unchecked
        let range = unsafe { Range::new_unchecked(10, 20) };
        assert_eq!(range.start(), 10);
        assert_eq!(range.last(), 20);
        assert_eq!(range.len(), 11);

        // Test with single element
        let single = unsafe { Range::new_unchecked(5, 5) };
        assert_eq!(single.len(), 1);
    }

    // --- RangeSet Mutation Tests ---

    #[test]
    fn test_rangeset_insert_basic() {
        let mut set = RangeSet::new();
        assert!(set.is_empty());

        // Insert first
        set.insert_range(&Range::new(10, 20));
        check_ranges(&set, &[(10, 20)]);
        assert_eq!(set.len(), 11);

        // Insert distinct after
        set.insert_range(&Range::new(30, 40));
        check_ranges(&set, &[(10, 20), (30, 40)]);

        // Insert distinct before
        set.insert_range(&Range::new(0, 5));
        check_ranges(&set, &[(0, 5), (10, 20), (30, 40)]);

        // Insert distinct between
        set.insert_range(&Range::new(22, 28));
        check_ranges(&set, &[(0, 5), (10, 20), (22, 28), (30, 40)]);
    }

    #[test]
    fn test_rangeset_insert_merge() {
        let mut set = make_set(&[(10, 20), (30, 40)]);

        // 1. Merge Left (Overlap)
        set.insert_range(&Range::new(15, 25));
        // 10..20 + 15..25 -> 10..25. (30..40 untouched)
        check_ranges(&set, &[(10, 25), (30, 40)]);

        // 2. Merge Right (Overlap)
        set.insert_range(&Range::new(28, 35));
        // 30..40 + 28..35 -> 28..40. (10..25 untouched)
        check_ranges(&set, &[(10, 25), (28, 40)]);

        // 3. Merge Bridge (Connect two ranges)
        set.insert_range(&Range::new(20, 30));
        // 10..25 + 28..40 + 20..30 covers the gap 25..28
        // Result should be 10..40
        check_ranges(&set, &[(10, 40)]);
    }

    #[test]
    fn test_rangeset_insert_adjacency() {
        let mut set = make_set(&[(10, 20)]);

        // Adjacent after
        set.insert_range(&Range::new(21, 25));
        check_ranges(&set, &[(10, 25)]);

        // Adjacent before
        set.insert_range(&Range::new(5, 9));
        check_ranges(&set, &[(5, 25)]);
    }

    #[test]
    fn test_rangeset_insert_contained() {
        let mut set = make_set(&[(10, 50)]);

        // Insert sub-range (should be no-op)
        assert!(!set.insert_range(&Range::new(20, 30)));
        check_ranges(&set, &[(10, 50)]);

        // Insert exact match
        assert!(!set.insert_range(&Range::new(10, 50)));
        check_ranges(&set, &[(10, 50)]);
    }

    #[test]
    fn test_rangeset_insert_consuming() {
        let mut set = make_set(&[(10, 20), (30, 40), (50, 60)]);

        // Insert huge range consuming everything
        set.insert_range(&Range::new(0, 100));
        check_ranges(&set, &[(0, 100)]);
    }

    #[test]
    fn test_rangeset_insert_n_at() {
        let mut set = RangeSet::new();
        set.insert_n_at(5, 10); // 10,11,12,13,14
        check_ranges(&set, &[(10, 14)]);

        set.insert_n_at(0, 20); // No-op
        check_ranges(&set, &[(10, 14)]);

        // Check overflow protection logic implies safe usage,
        // but here we just test valid big inputs if needed or standard usage.
        set.insert_n_at(1, 15); // Adjacent join
        check_ranges(&set, &[(10, 15)]);
    }

    // --- RangeSet Set Operations Tests ---

    #[test]
    fn test_rangeset_union() {
        let s1 = make_set(&[(0, 10), (20, 30)]);
        let s2 = make_set(&[(5, 25), (35, 40)]);

        // Expected: 0..10 U 5..25 -> 0..25
        // 0..25 U 20..30 -> 0..30
        // Result: 0..30, 35..40
        let u = s1.union(&s2);
        check_ranges(&u, &[(0, 30), (35, 40)]);

        // Commutative
        let u2 = s2.union(&s1);
        check_ranges(&u2, &[(0, 30), (35, 40)]);
    }

    #[test]
    fn test_rangeset_difference() {
        let a = make_set(&[(0, 50)]);

        // 1. Cut middle
        let b = make_set(&[(20, 30)]);
        let d1 = a.difference(&b);
        check_ranges(&d1, &[(0, 19), (31, 50)]);

        // 2. Cut start
        let c = make_set(&[(0, 10)]);
        let d2 = a.difference(&c);
        check_ranges(&d2, &[(11, 50)]);

        // 3. Cut end
        let d = make_set(&[(40, 60)]); // 40..60 overlaps 0..50 at 40..50
        let d3 = a.difference(&d);
        check_ranges(&d3, &[(0, 39)]);

        // 4. Multi-cut
        let e = make_set(&[(10, 15), (35, 40)]);
        let d4 = a.difference(&e);
        check_ranges(&d4, &[(0, 9), (16, 34), (41, 50)]);
    }

    #[test]
    fn test_rangeset_difference_complex() {
        let a = make_set(&[(0, 10), (20, 30), (40, 50)]);
        let b = make_set(&[(5, 25), (45, 55)]);

        // 0..10 - 5..25 -> 0..4
        // 20..30 - 5..25 -> 26..30
        // 40..50 - 45..55 -> 40..44
        let res = a.difference(&b);
        check_ranges(&res, &[(0, 4), (26, 30), (40, 44)]);
    }

    // --- FrozenRangeSet Tests ---

    #[test]
    fn test_frozen_lifecycle() {
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(0, 10));
        set.insert_range(&Range::new(20, 30));

        let frozen = set.freeze();
        assert_eq!(frozen.ranges_count(), 2);
        assert!(frozen.contains_n(5));
        assert!(frozen.contains(&Range::new(20, 25)));

        // Convert back
        let thawed: RangeSet = frozen.into();
        check_ranges(&thawed, &[(0, 10), (20, 30)]);
    }

    #[test]
    fn test_frozen_operators() {
        let s1 = make_set(&[(0, 10)]);
        let mut s2 = RangeSet::new();
        s2.insert_range(&Range::new(5, 15));
        let f2 = s2.freeze();

        // Union: s1 | &f2
        let u = &s1 | &f2;
        check_ranges(&u, &[(0, 15)]);

        // Difference: s1 - &f2
        let d = &s1 - &f2; // 0..10 - 5..15 -> 0..4
        check_ranges(&d, &[(0, 4)]);

        // Assign operators
        let mut s3 = s1.clone();
        s3 |= &f2;
        check_ranges(&s3, &[(0, 15)]);

        let mut s4 = s1;
        s4 -= &f2;
        check_ranges(&s4, &[(0, 4)]);
    }

    // --- Chunks Tests ---

    #[test]
    fn test_chunks_basic() {
        let mut set = make_set(&[(0, 9)]); // 10 items
        let chunks: Vec<FrozenRangeSet> = set.into_chunks(4).collect();

        // Expected chunks of size ~4
        // 0..9 -> [0..3] (4), [4..7] (4), [8..9] (2)
        assert_eq!(chunks.len(), 3);

        let c1 = &chunks[0];
        assert_eq!(c1.len(), 1);
        assert_eq!(c1[0], Range::new(0, 3));

        let c2 = &chunks[1];
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0], Range::new(4, 7));

        let c3 = &chunks[2];
        assert_eq!(c3.len(), 1);
        assert_eq!(c3[0], Range::new(8, 9));
    }

    #[test]
    fn test_chunks_fragmented() {
        let mut set = make_set(&[(0, 1), (10, 11), (20, 21)]); // Sizes: 2, 2, 2
        // Chunk size 3
        // Chunk 1: [0..1] (2), [10..10] (1) -> Total 3. Remainder of 10..11 is 11..11
        // Chunk 2: [11..11] (1), [20..21] (2) -> Total 3.
        let chunks: Vec<FrozenRangeSet> = set.into_chunks(3).collect();

        assert_eq!(chunks.len(), 2);

        // Verify Chunk 1
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[0][0], Range::new(0, 1));
        assert_eq!(chunks[0][1], Range::new(10, 10));

        // Verify Chunk 2
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[1][0], Range::new(11, 11));
        assert_eq!(chunks[1][1], Range::new(20, 21));
    }

    // --- HTTP Tests (Conditional) ---

    #[cfg(feature = "http")]
    #[test]
    fn test_http_parsing_success() {
        let total_size = 1000;

        // 1. Standard
        let s = RangeSet::parse_ranges_headers("bytes=0-499", total_size).unwrap();
        check_ranges(&s, &[(0, 499)]);

        // 2. Open end
        let s = RangeSet::parse_ranges_headers("bytes=500-", total_size).unwrap();
        check_ranges(&s, &[(500, 999)]);

        // 3. Suffix
        let s = RangeSet::parse_ranges_headers("bytes=-100", total_size).unwrap();
        // Last 100 bytes: 900..999
        check_ranges(&s, &[(900, 999)]);

        // 4. Multiple and Overlap clamping
        // 0-10, 5-20 -> 0-20. -10 -> 990-999.
        let s = RangeSet::parse_ranges_headers("bytes=0-10,5-20,-10", total_size).unwrap();
        check_ranges(&s, &[(0, 20), (990, 999)]);

        // 5. Clamping end
        let s = RangeSet::parse_ranges_headers("bytes=0-2000", total_size).unwrap();
        check_ranges(&s, &[(0, 999)]);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_http_parsing_errors() {
        let total = 100;

        // Start > Last
        assert_eq!(RangeSet::parse_ranges_headers("bytes=50-40", total), Err(Error::Invalid));

        // Start >= Total
        assert_eq!(RangeSet::parse_ranges_headers("bytes=100-", total), Err(Error::Invalid));

        // Empty entity
        assert_eq!(RangeSet::parse_ranges_headers("bytes=0-50", 0), Err(Error::Empty));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_http_generation() {
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(0, 9));
        set.insert_range(&Range::new(20, 29));

        let header = set.to_http_range_header().unwrap();
        assert_eq!(&*header, "bytes=0-9,20-29");

        // Frozen
        let frozen = set.freeze();
        let f_header = frozen.to_http_range_header().unwrap();
        assert_eq!(&*f_header, "bytes=0-9,20-29");
    }

    // --- Display/Debug Tests ---

    #[test]
    fn test_formatting() {
        // Range
        let r = Range::new(0, 1023); // 0 ~ 1023 B
        let s = format!("{}", r);
        assert!(s.contains("0 ~ 1023 B"));

        let r2 = Range::new(0, 2048); // 0 ~ 2 KiB
        let s2 = format!("{}", r2);
        assert!(s2.contains("2 KiB"));

        // RangeSet
        let set = make_set(&[(0, 10), (20, 30)]);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("RangeSet {0..=10, 20..=30}"));
    }

    #[test]
    fn test_rangeset_union_merge_explicit() {
        // 场景 1: 交错合并
        // s1: [0, 10], [20, 30], [40, 50]
        // s2: [5, 25], [35, 45]
        // 合并逻辑:
        // [0, 10] U [5, 25] -> [0, 25] (延伸)
        // [0, 25] U [20, 30] -> [0, 30] (继续延伸)
        // 下一个 gap 是 [30, 35] (空隙)
        // [35, 45] U [40, 50] -> [35, 50]
        // 结果: [0, 30], [35, 50]
        let s1 = make_set(&[(0, 10), (20, 30), (40, 50)]);
        let s2 = make_set(&[(5, 25), (35, 45)]);

        let res = s1.union_merge(&s2);
        check_ranges(&res, &[(0, 30), (35, 50)]);

        // 验证交换律
        let res2 = s2.union_merge(&s1);
        check_ranges(&res2, &[(0, 30), (35, 50)]);

        // 场景 2: 完全不重叠
        let d1 = make_set(&[(0, 5)]);
        let d2 = make_set(&[(10, 15)]);
        check_ranges(&d1.union_merge(&d2), &[(0, 5), (10, 15)]);

        // 场景 3: 刚好相邻 (Should merge)
        // 0..=5 (0,1,2,3,4,5) 和 6..=10 (6,7,8,9,10) 是相邻的
        let t1 = make_set(&[(0, 5)]);
        let t2 = make_set(&[(6, 10)]);
        check_ranges(&t1.union_merge(&t2), &[(0, 10)]);

        // 场景 4: 包含关系
        let large = make_set(&[(0, 100)]);
        let small = make_set(&[(20, 30), (50, 60)]);
        check_ranges(&large.union_merge(&small), &[(0, 100)]);
        check_ranges(&small.union_merge(&large), &[(0, 100)]);

        // 场景 5: 复杂覆盖
        // A: [0, 10], [100, 110]
        // B: [5, 105] (桥接两端)
        // 结果: [0, 110]
        let gap_set = make_set(&[(0, 10), (100, 110)]);
        let bridge_set = make_set(&[(5, 105)]);
        check_ranges(&gap_set.union_merge(&bridge_set), &[(0, 110)]);
    }

    // --- RangeSet Special Operations Tests ---

    #[test]
    fn test_rangeset_union_assign() {
        let mut set1 = make_set(&[(0, 10), (20, 30)]);
        let set2 = make_set(&[(5, 15), (25, 35)]);

        set1.union_assign(&set2);
        // Note: union_assign may not fully merge all ranges if there's a gap
        // (0,10) U (5,15) = (0,15), (20,30) U (25,35) = (20,35)
        // But there's still a gap between 15 and 20
        check_ranges(&set1, &[(0, 15), (20, 35)]);

        // Test with empty set
        let mut set3 = RangeSet::new();
        let set4 = make_set(&[(0, 10)]);
        set3.union_assign(&set4);
        check_ranges(&set3, &[(0, 10)]);

        // Test assigning empty to non-empty
        let mut set5 = make_set(&[(0, 10)]);
        let set6 = RangeSet::new();
        set5.union_assign(&set6);
        check_ranges(&set5, &[(0, 10)]);
    }

    #[test]
    fn test_rangeset_difference_assign() {
        let mut set1 = make_set(&[(0, 50)]);
        let set2 = make_set(&[(20, 30)]);

        set1.difference_assign(&set2);
        check_ranges(&set1, &[(0, 19), (31, 50)]);

        // Test with empty sets
        let mut set3 = make_set(&[(0, 10)]);
        let set4 = RangeSet::new();
        set3.difference_assign(&set4);
        check_ranges(&set3, &[(0, 10)]);

        let mut set5 = RangeSet::new();
        let set6 = make_set(&[(0, 10)]);
        set5.difference_assign(&set6);
        assert!(set5.is_empty());
    }

    #[test]
    fn test_rangeset_frozen_operations() {
        let set1 = make_set(&[(0, 10)]);
        let set2 = make_set(&[(5, 15)]);
        let frozen2 = set2.freeze();

        // union_frozen
        let result = set1.union_frozen(&frozen2);
        check_ranges(&result, &[(0, 15)]);

        // Test with empty RangeSet
        let empty = RangeSet::new();
        let result2 = empty.union_frozen(&frozen2);
        check_ranges(&result2, &[(5, 15)]);

        // Test with empty FrozenRangeSet
        let frozen_empty = RangeSet::new().freeze();
        let result3 = set1.union_frozen(&frozen_empty);
        check_ranges(&result3, &[(0, 10)]);

        // union_assign_frozen
        let mut set3 = make_set(&[(0, 10)]);
        set3.union_assign_frozen(&frozen2);
        check_ranges(&set3, &[(0, 15)]);

        // difference_frozen
        let set4 = make_set(&[(0, 20)]);
        let frozen3 = make_set(&[(5, 15)]).freeze();
        let result4 = set4.difference_frozen(&frozen3);
        check_ranges(&result4, &[(0, 4), (16, 20)]);

        // difference_assign_frozen
        let mut set5 = make_set(&[(0, 20)]);
        set5.difference_assign_frozen(&frozen3);
        check_ranges(&set5, &[(0, 4), (16, 20)]);
    }

    #[test]
    fn test_rangeset_from_iterator() {
        // From Range iterator
        let ranges = vec![Range::new(0, 10), Range::new(20, 30), Range::new(5, 15)];
        let set: RangeSet = ranges.into_iter().collect();
        // FromIterator inserts ranges one by one, merging adjacent/overlapping
        // (0,10) then (20,30) then (5,15) which merges with (0,10) to make (0,15)
        // There's still a gap between 15 and 20
        check_ranges(&set, &[(0, 15), (20, 30)]);

        // From tuple iterator
        let tuples = vec![(0, 10), (20, 30)];
        let set2: RangeSet = tuples.into_iter().collect();
        check_ranges(&set2, &[(0, 10), (20, 30)]);

        // Empty iterator
        let empty: Vec<Range> = vec![];
        let set3: RangeSet = empty.into_iter().collect();
        assert!(set3.is_empty());
    }

    #[test]
    fn test_rangeset_operators() {
        let set1 = make_set(&[(0, 10)]);
        let set2 = make_set(&[(5, 15)]);

        // BitOr (|)
        let union = &set1 | &set2;
        check_ranges(&union, &[(0, 15)]);

        // Sub (-)
        let diff = &set1 - &set2;
        check_ranges(&diff, &[(0, 4)]);

        // BitOrAssign (|=)
        let mut set3 = set1.clone();
        set3 |= &set2;
        check_ranges(&set3, &[(0, 15)]);

        // SubAssign (-=)
        let mut set4 = set1;
        set4 -= &set2;
        check_ranges(&set4, &[(0, 4)]);
    }

    #[test]
    fn test_rangeset_empty_operations() {
        let empty = RangeSet::new();
        let non_empty = make_set(&[(0, 10)]);

        // Empty union
        let result1 = empty.union(&non_empty);
        check_ranges(&result1, &[(0, 10)]);

        let result2 = non_empty.union(&empty);
        check_ranges(&result2, &[(0, 10)]);

        // Empty difference
        let result3 = empty.difference(&non_empty);
        assert!(result3.is_empty());

        let result4 = non_empty.difference(&empty);
        check_ranges(&result4, &[(0, 10)]);

        // Empty queries
        assert_eq!(empty.start(), None);
        assert_eq!(empty.last(), None);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.ranges_count(), 0);
        assert!(!empty.contains_n(5));
        assert!(!empty.contains(&Range::new(0, 10)));
    }

    #[test]
    fn test_frozen_rangeset_empty() {
        let empty = RangeSet::new().freeze();

        assert_eq!(empty.start(), None);
        assert_eq!(empty.last(), None);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.ranges_count(), 0);
        assert!(empty.is_empty());
        assert!(!empty.contains_n(5));
        assert!(!empty.contains(&Range::new(0, 10)));

        // Empty to HTTP header
        #[cfg(feature = "http")]
        assert!(empty.to_http_range_header().is_none());
    }

    #[test]
    fn test_rangeset_partial_eq() {
        let set1 = make_set(&[(0, 10)]);
        let range1 = Range::new(0, 10);

        // RangeSet == Range
        assert_eq!(set1, range1);
        assert_eq!(range1, set1);

        // Multiple ranges != single range
        let set2 = make_set(&[(0, 5), (7, 10)]);
        assert_ne!(set2, range1);

        // FrozenRangeSet comparisons
        let frozen = set1.freeze();
        assert_eq!(frozen, range1);
        assert_eq!(range1, frozen);
        assert_eq!(frozen, set1);
        assert_eq!(set1, frozen);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_http_edge_cases() {
        // Suffix range with 0 (returns Header error from underlying library)
        let result = RangeSet::parse_ranges_headers("bytes=-0", 100);
        assert!(result.is_err());

        // Suffix larger than total size
        let result = RangeSet::parse_ranges_headers("bytes=-200", 100).unwrap();
        check_ranges(&result, &[(0, 99)]); // Should saturate to full range

        // Empty RangeSet to HTTP header
        let empty = RangeSet::new();
        assert!(empty.to_http_range_header().is_none());

        // Single byte range
        let result = RangeSet::parse_ranges_headers("bytes=0-0", 100).unwrap();
        check_ranges(&result, &[(0, 0)]);

        // Multiple overlapping ranges should merge
        let result = RangeSet::parse_ranges_headers("bytes=0-10,5-15,20-30", 100).unwrap();
        check_ranges(&result, &[(0, 15), (20, 30)]);
    }

    #[test]
    fn test_display_formatting() {
        // Test Range display with different sizes
        let small = Range::new(0, 999);
        let display = format!("{}", small);
        assert!(display.contains("B")); // Should show in bytes

        // Test binary vs SI units
        let binary = Range::new(0, 1024);
        let binary_display = format!("{}", binary);
        assert!(binary_display.contains("KiB") || binary_display.contains("KB"));

        let si = Range::new(0, 1024);
        let si_display = format!("{:#}", si); // alternate format for SI
        assert!(si_display.contains("B"));

        // Test RangeSet Display
        let set = make_set(&[(0, 10), (20, 30)]);
        let display = format!("{}", set);
        assert!(display.contains("0"));
        assert!(display.contains("10"));
        assert!(display.contains("20"));
        assert!(display.contains("30"));
    }

    #[test]
    fn test_chunks_edge_cases() {
        // Empty set
        let mut empty = RangeSet::new();
        let chunks: Vec<FrozenRangeSet> = empty.into_chunks(10).collect();
        assert!(chunks.is_empty());

        // Single range smaller than block size
        let mut small = make_set(&[(0, 5)]);
        let chunks: Vec<FrozenRangeSet> = small.into_chunks(10).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0][0], Range::new(0, 5));

        // Single range larger than block size
        let mut large = make_set(&[(0, 25)]);
        let chunks: Vec<FrozenRangeSet> = large.into_chunks(10).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0][0], Range::new(0, 9));
        assert_eq!(chunks[1][0], Range::new(10, 19));
        assert_eq!(chunks[2][0], Range::new(20, 25));

        // Block size of 1
        let mut set = make_set(&[(0, 2)]);
        let chunks: Vec<FrozenRangeSet> = set.into_chunks(1).collect();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_insert_range_return_value() {
        let mut set = RangeSet::new();

        // First insert should return true
        assert!(set.insert_range(&Range::new(0, 10)));

        // Inserting same range should return false
        assert!(!set.insert_range(&Range::new(0, 10)));

        // Inserting subset should return false
        assert!(!set.insert_range(&Range::new(2, 8)));

        // Inserting overlapping should return true
        assert!(set.insert_range(&Range::new(5, 15)));
        check_ranges(&set, &[(0, 15)]);
    }

    // --- Edge Case and Boundary Tests ---

    #[test]
    fn test_range_boundary_values() {
        // Test with maximum valid value (usize::MAX - 1)
        let max_range = Range::new(0, usize::MAX - 1);
        assert_eq!(max_range.start(), 0);
        assert_eq!(max_range.last(), usize::MAX - 1);
        assert_eq!(max_range.len(), usize::MAX);

        // Test with high values near MAX
        let high_range = Range::new(usize::MAX - 10, usize::MAX - 1);
        assert_eq!(high_range.len(), 10);
        assert!(high_range.contains_n(usize::MAX - 5));
        assert!(!high_range.contains_n(usize::MAX));

        // Single point at MAX - 1
        let single_max = Range::new(usize::MAX - 1, usize::MAX - 1);
        assert_eq!(single_max.len(), 1);
        assert!(single_max.contains_n(usize::MAX - 1));

        // Zero range (single point)
        let zero_range = Range::new(0, 0);
        assert_eq!(zero_range.len(), 1);
        assert_eq!(zero_range.start(), 0);
        assert_eq!(zero_range.last(), 0);
    }

    #[test]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_with_usize_max() {
        // This should panic because last cannot be usize::MAX
        let _ = Range::new(0, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_both_usize_max() {
        // This should panic
        let _ = Range::new(usize::MAX, usize::MAX);
    }

    #[test]
    fn test_range_operations_with_boundary_values() {
        let r1 = Range::new(usize::MAX - 20, usize::MAX - 10);
        let r2 = Range::new(usize::MAX - 15, usize::MAX - 5);

        // Intersection
        let intersection = r1.intersection(&r2).unwrap();
        assert_eq!(intersection.start(), usize::MAX - 15);
        assert_eq!(intersection.last(), usize::MAX - 10);

        // Union (overlapping)
        let union = r1.union(&r2).unwrap();
        assert_eq!(union.start(), usize::MAX - 20);
        assert_eq!(union.last(), usize::MAX - 5);

        // Adjacent ranges near MAX
        let r3 = Range::new(usize::MAX - 10, usize::MAX - 6);
        let r4 = Range::new(usize::MAX - 5, usize::MAX - 1);
        assert!(r3.is_adjacent(&r4));

        // Midpoint with large values
        let large = Range::new(usize::MAX - 100, usize::MAX - 1);
        let mid = large.midpoint();
        assert!((usize::MAX - 100..=usize::MAX - 1).contains(&mid));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_insert_n_at_overflow() {
        let mut set = RangeSet::new();
        // This should panic due to overflow: usize::MAX - 1 + 100 overflows
        // Note: This only panics in debug mode due to assert!
        set.insert_n_at(100, usize::MAX - 1);
    }

    #[test]
    fn test_insert_n_at_boundary() {
        let mut set = RangeSet::new();

        // Insert at zero with zero length (should be no-op)
        set.insert_n_at(0, 0);
        assert!(set.is_empty());

        // Insert single element at MAX - 2 (MAX - 1 would overflow to MAX)
        set.insert_n_at(1, usize::MAX - 2);
        check_ranges(&set, &[(usize::MAX - 2, usize::MAX - 2)]);

        // Insert at high value
        let mut set2 = RangeSet::new();
        set2.insert_n_at(10, usize::MAX - 20);
        check_ranges(&set2, &[(usize::MAX - 20, usize::MAX - 11)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_chunks_zero_block_size() {
        let mut set = make_set(&[(0, 10)]);
        // This should panic in debug mode because block_size must be > 0
        let _ = set.into_chunks(0).collect::<Vec<_>>();
    }

    #[test]
    fn test_rangeset_with_maximum_values() {
        let mut set = RangeSet::new();

        // Insert range at the high end
        set.insert_range(&Range::new(usize::MAX - 100, usize::MAX - 50));
        assert_eq!(set.len(), 51);
        assert!(set.contains_n(usize::MAX - 75));
        assert!(!set.contains_n(usize::MAX - 49));

        // Insert another high range
        set.insert_range(&Range::new(usize::MAX - 40, usize::MAX - 1));
        assert_eq!(set.ranges_count(), 2);

        // Merge them
        set.insert_range(&Range::new(usize::MAX - 50, usize::MAX - 40));
        check_ranges(&set, &[(usize::MAX - 100, usize::MAX - 1)]);
    }

    #[test]
    fn test_rangeset_contains_with_boundaries() {
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(0, 10));
        set.insert_range(&Range::new(usize::MAX - 10, usize::MAX - 1));

        // Test contains_n at boundaries
        assert!(set.contains_n(0));
        assert!(set.contains_n(10));
        assert!(!set.contains_n(11));

        assert!(set.contains_n(usize::MAX - 10));
        assert!(set.contains_n(usize::MAX - 1));
        assert!(!set.contains_n(usize::MAX - 11));

        // Test contains Range at boundaries
        assert!(set.contains(&Range::new(0, 10)));
        assert!(set.contains(&Range::new(0, 5)));
        assert!(!set.contains(&Range::new(0, 11)));

        assert!(set.contains(&Range::new(usize::MAX - 10, usize::MAX - 1)));
        assert!(!set.contains(&Range::new(usize::MAX - 11, usize::MAX - 1)));
    }

    #[test]
    fn test_frozen_with_extreme_values() {
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(0, 0));
        set.insert_range(&Range::new(usize::MAX / 2, usize::MAX / 2 + 100));
        set.insert_range(&Range::new(usize::MAX - 1, usize::MAX - 1));

        let frozen = set.freeze();

        assert_eq!(frozen.start(), Some(0));
        assert_eq!(frozen.last(), Some(usize::MAX - 1));
        assert_eq!(frozen.ranges_count(), 3);

        assert!(frozen.contains_n(0));
        assert!(frozen.contains_n(usize::MAX / 2 + 50));
        assert!(frozen.contains_n(usize::MAX - 1));
        assert!(!frozen.contains_n(1));
    }

    #[test]
    fn test_union_with_extreme_gaps() {
        // Test union with very large gaps between ranges
        let set1 = make_set(&[(0, 10)]);
        let set2 = make_set(&[(usize::MAX - 10, usize::MAX - 1)]);

        let union = set1.union(&set2);
        assert_eq!(union.ranges_count(), 2);
        check_ranges(&union, &[(0, 10), (usize::MAX - 10, usize::MAX - 1)]);

        // Test union_merge with extreme values
        let merged = set1.union_merge(&set2);
        check_ranges(&merged, &[(0, 10), (usize::MAX - 10, usize::MAX - 1)]);
    }

    #[test]
    fn test_difference_with_extreme_values() {
        // Large range minus small range at boundaries
        let large = make_set(&[(0, usize::MAX - 1)]);
        let small = make_set(&[(usize::MAX / 2 - 5, usize::MAX / 2 + 5)]);

        let diff = large.difference(&small);
        assert_eq!(diff.ranges_count(), 2);
        assert_eq!(diff.start(), Some(0));
        assert_eq!(diff.last(), Some(usize::MAX - 1));

        // Verify the hole exists
        assert!(!diff.contains_n(usize::MAX / 2));
        assert!(diff.contains_n(usize::MAX / 2 - 6));
        assert!(diff.contains_n(usize::MAX / 2 + 6));
    }

    #[test]
    fn test_chunks_with_extreme_ranges() {
        // Test chunking a range near MAX
        let mut set = make_set(&[(usize::MAX - 100, usize::MAX - 1)]);
        let chunks: Vec<_> = set.into_chunks(25).collect();

        // Should create 4 chunks of 25 elements each
        assert_eq!(chunks.len(), 4);

        // Verify total coverage - need to count actual elements, not range count
        let total: usize = chunks.iter().map(|c| c.iter().map(|r| r.len()).sum::<usize>()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_range_edge_case_operations() {
        // Test adjacent at zero
        let r1 = Range::new(0, 5);
        let r2 = Range::new(6, 10);
        assert!(r1.is_adjacent(&r2));
        assert!(r2.is_adjacent(&r1));

        // Test not adjacent with gap at zero
        let r3 = Range::new(0, 3);
        let r4 = Range::new(5, 10);
        assert!(!r3.is_adjacent(&r4));

        // Test intersects at single point
        let r5 = Range::new(0, 10);
        let r6 = Range::new(10, 20);
        assert!(r5.intersects(&r6));

        // Test difference resulting in empty
        let r7 = Range::new(5, 10);
        let r8 = Range::new(5, 10);
        assert_eq!(r7.difference(&r8), (None, None));

        // Test difference at boundaries
        let r9 = Range::new(0, 10);
        let r10 = Range::new(0, 0);
        let (left, right) = r9.difference(&r10);
        assert_eq!(left, None);
        assert_eq!(right, Some(Range::new(1, 10)));
    }

    #[test]
    fn test_rangeset_stress_many_ranges() {
        // Insert many small ranges
        let mut set = RangeSet::new();
        for i in (0..1000).step_by(2) {
            set.insert_range(&Range::new(i, i));
        }

        assert_eq!(set.ranges_count(), 500);
        assert_eq!(set.len(), 500);

        // Merge them all
        for i in (1..1000).step_by(2) {
            set.insert_range(&Range::new(i, i));
        }

        assert_eq!(set.ranges_count(), 1);
        check_ranges(&set, &[(0, 999)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_range_new_unchecked_panic_order() {
        // In debug mode, new_unchecked should panic if start > last
        let _ = unsafe { Range::new_unchecked(20, 10) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_unchecked_panic_max() {
        // In debug mode, new_unchecked should panic if last == usize::MAX
        let _ = unsafe { Range::new_unchecked(0, usize::MAX) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_insert_n_at_unchecked_overflow() {
        let mut set = RangeSet::new();
        // In debug mode, this should panic due to overflow
        unsafe { set.insert_n_at_unchecked(100, usize::MAX - 1) };
    }

    #[test]
    fn test_range_len_calculation() {
        // Verify len calculation doesn't overflow
        let r = Range::new(0, usize::MAX - 1);
        assert_eq!(r.len(), usize::MAX);

        let r2 = Range::new(usize::MAX - 10, usize::MAX - 1);
        assert_eq!(r2.len(), 10);
    }

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_range_from_tuple_invalid_order() {
        // Should panic when start > last
        let _ = Range::from((100, 50));
    }

    #[test]
    fn test_range_contains_boundary() {
        let r = Range::new(10, 20);

        // Test exact boundaries
        assert!(r.contains_n(10));
        assert!(r.contains_n(20));

        // Test just outside boundaries
        assert!(!r.contains_n(9));
        assert!(!r.contains_n(21));

        // Test with usize boundaries
        let r2 = Range::new(0, usize::MAX - 1);
        assert!(r2.contains_n(0));
        assert!(r2.contains_n(usize::MAX - 1));
        assert!(!r2.contains_n(usize::MAX));
    }

    #[test]
    fn test_rangeset_insert_range_boundary_merge() {
        let mut set = RangeSet::new();

        // Insert ranges that should merge at boundaries
        set.insert_range(&Range::new(0, 5));
        set.insert_range(&Range::new(6, 10)); // Adjacent, should merge
        check_ranges(&set, &[(0, 10)]);

        // Insert range that extends exactly to boundary
        set.insert_range(&Range::new(11, 20));
        check_ranges(&set, &[(0, 20)]);

        // Insert range at high boundary
        let mut set2 = RangeSet::new();
        set2.insert_range(&Range::new(usize::MAX - 10, usize::MAX - 5));
        set2.insert_range(&Range::new(usize::MAX - 4, usize::MAX - 1));
        check_ranges(&set2, &[(usize::MAX - 10, usize::MAX - 1)]);
    }

    #[test]
    fn test_range_intersection_no_overlap() {
        let r1 = Range::new(0, 10);
        let r2 = Range::new(11, 20);

        assert!(r1.intersection(&r2).is_none());
        assert!(r2.intersection(&r1).is_none());

        // Test at boundaries
        let r3 = Range::new(0, 10);
        let r4 = Range::new(10, 20);
        assert!(r3.intersection(&r4).is_some()); // They touch at 10
    }

    #[test]
    fn test_range_union_non_adjacent() {
        let r1 = Range::new(0, 10);
        let r2 = Range::new(12, 20); // Gap at 11

        assert!(r1.union(&r2).is_none());
        assert!(r2.union(&r1).is_none());

        // Test adjacent
        let r3 = Range::new(0, 10);
        let r4 = Range::new(11, 20);
        assert!(r3.union(&r4).is_some());
    }

    #[test]
    fn test_rangeset_difference_no_overlap() {
        let set1 = make_set(&[(0, 10)]);
        let set2 = make_set(&[(20, 30)]);

        let diff = set1.difference(&set2);
        check_ranges(&diff, &[(0, 10)]); // No change

        let diff2 = set2.difference(&set1);
        check_ranges(&diff2, &[(20, 30)]); // No change
    }

    #[test]
    fn test_rangeset_union_empty_combinations() {
        let empty = RangeSet::new();
        let non_empty = make_set(&[(0, 10)]);

        // Empty | Empty = Empty
        let result = empty.union(&empty);
        assert!(result.is_empty());

        // Empty | NonEmpty = NonEmpty
        let result = empty.union(&non_empty);
        check_ranges(&result, &[(0, 10)]);

        // NonEmpty | Empty = NonEmpty
        let result = non_empty.union(&empty);
        check_ranges(&result, &[(0, 10)]);
    }

    #[test]
    fn test_frozen_start_last_none() {
        let empty = RangeSet::new().freeze();
        assert_eq!(empty.start(), None);
        assert_eq!(empty.last(), None);

        // Non-empty
        let non_empty = make_set(&[(5, 15), (20, 30)]).freeze();
        assert_eq!(non_empty.start(), Some(5));
        assert_eq!(non_empty.last(), Some(30));
    }

    #[test]
    fn test_rangeset_contains_edge_cases() {
        let mut set = RangeSet::new();

        // Empty set contains nothing
        assert!(!set.contains_n(0));
        assert!(!set.contains(&Range::new(0, 10)));

        // Single point
        set.insert_range(&Range::new(5, 5));
        assert!(set.contains_n(5));
        assert!(!set.contains_n(4));
        assert!(!set.contains_n(6));
        assert!(set.contains(&Range::new(5, 5)));
        assert!(!set.contains(&Range::new(4, 5)));
        assert!(!set.contains(&Range::new(5, 6)));
    }

    #[test]
    fn test_range_midpoint_edge_cases() {
        // Single point
        let r = Range::new(5, 5);
        assert_eq!(r.midpoint(), 5);

        // Two points
        let r = Range::new(5, 6);
        assert_eq!(r.midpoint(), 5); // Rounds down

        // Large values
        let r = Range::new(usize::MAX - 100, usize::MAX - 1);
        let mid = r.midpoint();
        assert!((usize::MAX - 100..=usize::MAX - 1).contains(&mid));
        assert_eq!(mid, usize::MAX - 100 + 49); // (MAX-100 + MAX-1) / 2
    }

    #[test]
    fn test_rangeset_multiple_operations_chain() {
        // Test a chain of operations
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(0, 10));
        set.insert_range(&Range::new(20, 30));

        let set2 = make_set(&[(5, 25)]);

        // Union then difference
        let union = set.union(&set2);
        check_ranges(&union, &[(0, 30)]);

        let set3 = make_set(&[(8, 12)]);
        let diff = union.difference(&set3);
        check_ranges(&diff, &[(0, 7), (13, 30)]);

        // Freeze and convert back
        let frozen = diff.freeze();
        let thawed: RangeSet = frozen.into();
        check_ranges(&thawed, &[(0, 7), (13, 30)]);
    }

    #[test]
    fn test_range_is_empty_always_false() {
        // Range is never empty by definition
        let r1 = Range::new(0, 0);
        assert!(!r1.is_empty());

        let r2 = Range::new(10, 20);
        assert!(!r2.is_empty());

        let r3 = Range::new(usize::MAX - 1, usize::MAX - 1);
        assert!(!r3.is_empty());
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_http_with_boundary_values() {
        let total_size = usize::MAX / 2;

        // Request near the maximum
        let result = RangeSet::parse_ranges_headers(&format!("bytes=0-{}", total_size - 1), total_size).unwrap();
        assert_eq!(result.len(), total_size);

        // Request last byte only
        let result =
            RangeSet::parse_ranges_headers(&format!("bytes={}-{}", total_size - 1, total_size - 1), total_size)
                .unwrap();
        check_ranges(&result, &[(total_size - 1, total_size - 1)]);

        // Request suffix of entire size
        let result = RangeSet::parse_ranges_headers(&format!("bytes=-{}", total_size), total_size).unwrap();
        check_ranges(&result, &[(0, total_size - 1)]);
    }

    // --- Additional Panic Tests for Assertions ---

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_range_new_panic_reversed_large_values() {
        // Test with large reversed values
        let _ = Range::new(usize::MAX - 1, usize::MAX - 10);
    }

    #[test]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_panic_last_is_max_zero_start() {
        // Test with start = 0, last = MAX
        let _ = Range::new(0, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_range_new_panic_last_is_max_large_start() {
        // Test with large start, last = MAX
        let _ = Range::new(usize::MAX - 100, usize::MAX);
    }

    #[test]
    #[should_panic]
    fn test_insert_n_at_panic_overflow_exact_max() {
        let mut set = RangeSet::new();
        // at + n would equal usize::MAX exactly
        set.insert_n_at(2, usize::MAX - 1);
    }

    #[test]
    #[should_panic]
    fn test_insert_n_at_panic_overflow_exceed_max() {
        let mut set = RangeSet::new();
        // at + n would exceed usize::MAX
        set.insert_n_at(10, usize::MAX - 5);
    }

    #[test]
    #[should_panic]
    fn test_insert_n_at_panic_large_n_at_boundary() {
        let mut set = RangeSet::new();
        // Large n at high boundary
        set.insert_n_at(usize::MAX / 2, usize::MAX / 2 + 1);
    }

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_from_tuple_panic_small_difference() {
        // Test with small but reversed values
        let _ = Range::from((2, 1));
    }

    #[test]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_from_tuple_panic_zero_vs_max() {
        // Test with maximum difference
        let _ = Range::from((usize::MAX - 1, 0));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "last must be less than usize::MAX")]
    fn test_new_unchecked_panic_last_equals_max_in_debug() {
        // Debug assertion should catch this
        let _ = unsafe { Range::new_unchecked(usize::MAX - 10, usize::MAX) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "start must be less than or equal to last")]
    fn test_new_unchecked_panic_reversed_in_debug() {
        // Debug assertion should catch this
        let _ = unsafe { Range::new_unchecked(100, 50) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_insert_n_at_unchecked_panic_overflow_in_debug() {
        let mut set = RangeSet::new();
        // Debug assertion should catch overflow
        unsafe {
            set.insert_n_at_unchecked(100, usize::MAX - 50);
        }
    }

    #[test]
    fn test_range_try_from_empty_range() {
        // Empty range starting at 0 should return error (underflow)
        let empty = 0..0;
        assert!(Range::try_from(&empty).is_err());

        // Note: Empty ranges like 100..100 will panic in Range::new
        // because after subtracting 1 from end, we get last < start
        // This is expected behavior - use try_from carefully
    }

    #[test]
    fn test_range_operations_no_panic_with_valid_boundaries() {
        // These should NOT panic - they are valid operations
        let r1 = Range::new(0, usize::MAX - 1);
        assert_eq!(r1.len(), usize::MAX);
        assert!(r1.contains_n(0));
        assert!(r1.contains_n(usize::MAX - 1));

        let r2 = Range::new(usize::MAX - 2, usize::MAX - 1);
        assert_eq!(r2.len(), 2);
        assert_eq!(r2.midpoint(), usize::MAX - 2);

        // Operations should work
        let r3 = Range::new(0, 100);
        let r4 = Range::new(50, usize::MAX - 1);
        assert!(r3.intersects(&r4));
        let intersection = r3.intersection(&r4).unwrap();
        assert_eq!(intersection.start(), 50);
        assert_eq!(intersection.last(), 100);
    }

    #[test]
    fn test_rangeset_operations_no_panic_near_max() {
        // These should NOT panic - they are valid operations near boundaries
        let mut set = RangeSet::new();
        set.insert_range(&Range::new(usize::MAX - 100, usize::MAX - 50));
        set.insert_range(&Range::new(usize::MAX - 49, usize::MAX - 1));

        // First range: MAX-100 to MAX-50 = 51 elements
        // Second range: MAX-49 to MAX-1 = 51 elements
        // They are adjacent and should merge
        assert_eq!(set.ranges_count(), 1);
        assert_eq!(set.len(), 100); // 51 + 51 - 2 = 100 (they share boundary)
        assert!(set.contains_n(usize::MAX - 75));
        assert!(set.contains_n(usize::MAX - 1));

        // Verify the merged range
        check_ranges(&set, &[(usize::MAX - 100, usize::MAX - 1)]);

        // Freeze should work
        let frozen = set.freeze();
        assert_eq!(frozen.ranges_count(), 1); // One merged range
        let total_elements: usize = frozen.iter().map(|r| r.len()).sum();
        assert_eq!(total_elements, 100);
        assert!(frozen.contains_n(usize::MAX - 50));
    }

    #[test]
    fn test_insert_n_at_valid_at_near_max() {
        // These should NOT panic - they are valid operations
        let mut set = RangeSet::new();

        // Valid: inserting 1 element at MAX-2 creates range [MAX-2, MAX-2]
        set.insert_n_at(1, usize::MAX - 2);
        check_ranges(&set, &[(usize::MAX - 2, usize::MAX - 2)]);

        // Valid: inserting 10 elements at MAX-20
        let mut set2 = RangeSet::new();
        set2.insert_n_at(10, usize::MAX - 20);
        check_ranges(&set2, &[(usize::MAX - 20, usize::MAX - 11)]);

        // Valid: maximum safe insertion
        let mut set3 = RangeSet::new();
        set3.insert_n_at(usize::MAX - 100, 0);
        check_ranges(&set3, &[(0, usize::MAX - 101)]);
    }
}
