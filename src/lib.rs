#![feature(btree_cursors)]
use std::{
    collections::BTreeMap,
    fmt,
    ops::{self, BitOr, BitOrAssign, Bound, Sub, SubAssign},
};
use thiserror::Error;

/// An inclusive range defined by start and last offsets.
///
/// This struct represents a contiguous range of unsigned integers where both
/// the start and end points are included in the range. It provides various
/// utility methods for manipulating and querying ranges.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
pub struct OffsetRange {
    start: usize,
    last: usize,
}

impl OffsetRange {
    /// Creates a new range with the given start and last values.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting offset (inclusive)
    /// * `last` - The ending offset (inclusive)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `start` > `last`.
    #[inline]
    pub fn new(start: usize, last: usize) -> Self {
        debug_assert!(start <= last);
        OffsetRange { start, last }
    }

    /// Returns the start offset of the range.
    #[inline]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the last offset of the range.
    #[inline]
    pub fn last(&self) -> usize {
        self.last
    }

    /// Returns the length of the range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::OffsetRange;
    /// let range = OffsetRange::new(5, 10);
    /// assert_eq!(range.len(), 6);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.last - self.start + 1
    }

    /// Always returns `false` because an `OffsetRange` is never empty.
    ///
    /// An `OffsetRange` is always considered non-empty because it represents
    /// an inclusive range from start to last where both ends are included.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

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
    pub fn contains_n(&self, n: usize) -> bool {
        self.start <= n && n <= self.last
    }

    /// Checks if the range contains another range.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to check for containment
    ///
    /// # Returns
    ///
    /// `true` if [other.start, other.last] is completely within [self.start, self.last].
    #[inline]
    pub fn contains(&self, other: &Self) -> bool {
        self.start <= other.start && self.last >= other.last
    }

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
    pub fn intersects(&self, other: &Self) -> bool {
        self.start <= other.last && self.last >= other.start
    }

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
    fn intersects_or_adjacent(&self, other: &Self) -> bool {
        self.start.saturating_sub(1) <= other.last && other.start.saturating_sub(1) <= self.last
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
    pub fn merge(&self, other: &Self) -> Option<Self> {
        self.intersects_or_adjacent(other).then_some({
            let start = self.start.min(other.start);
            let last = self.last.max(other.last);
            OffsetRange::new(start, last)
        })
    }
}

impl TryFrom<&ops::Range<usize>> for OffsetRange {
    type Error = Error;

    /// Attempts to create an `OffsetRange` from a standard library range.
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
    /// Returns an error if the range end would cause an overflow when converted
    /// to an inclusive range (e.g., when `end` is 0 and we try to compute `end - 1`).
    #[inline]
    fn try_from(rng: &ops::Range<usize>) -> Result<Self, Self::Error> {
        let start = rng.start;
        let last = rng.end.checked_sub(1).ok_or(Error::IndexOverflow)?;
        Ok(OffsetRange::new(start, last))
    }
}

impl From<&ops::RangeInclusive<usize>> for OffsetRange {
    /// Creates an `OffsetRange` from a reference to an inclusive range.
    ///
    /// # Arguments
    ///
    /// * `rng` - The inclusive range to convert
    #[inline]
    fn from(rng: &ops::RangeInclusive<usize>) -> Self {
        Self {
            start: *rng.start(),
            last: *rng.end(),
        }
    }
}

impl From<(usize, usize)> for OffsetRange {
    /// Creates an `OffsetRange` from a tuple of (start, last).
    ///
    /// # Arguments
    ///
    /// * `rng` - A tuple where the first element is the start and the second is the last
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the start value is greater than the last value.
    #[inline]
    fn from(rng: (usize, usize)) -> Self {
        debug_assert!(rng.0 <= rng.1);
        Self {
            start: rng.0,
            last: rng.1,
        }
    }
}

impl fmt::Debug for OffsetRangeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut set_builder = f.debug_set();
        for (&start, &last) in &self.0 {
            set_builder.entry(&(start..=last));
        }
        set_builder.finish()
    }
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
/// # use sparse_ranges::{OffsetRange, OffsetRangeSet};
/// let mut set = OffsetRangeSet::new();
/// set.insert_range(&OffsetRange::new(0, 5));
/// set.insert_range(&OffsetRange::new(10, 15));
/// // The set now contains two separate ranges
/// ```
#[derive(Default, Clone)]
pub struct OffsetRangeSet(BTreeMap<usize, usize>);

impl OffsetRangeSet {
    /// Creates a new, empty `OffsetRangeSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::OffsetRangeSet;
    /// let set = OffsetRangeSet::new();
    /// assert!(set.is_empty());
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns the total number of offsets covered by all ranges in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{OffsetRange, OffsetRangeSet};
    /// let mut set = OffsetRangeSet::new();
    /// set.insert_range(&OffsetRange::new(0, 5));  // 6 offsets
    /// set.insert_range(&OffsetRange::new(10, 12)); // 3 offsets
    /// assert_eq!(set.len(), 9);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.0.iter().map(|(start, last)| last - start + 1).sum()
    }

    /// Checks if the set is empty.
    ///
    /// # Returns
    ///
    /// `true` if the set contains no ranges, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Checks if the set contains a specific offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset to check for containment
    ///
    /// # Returns
    ///
    /// `true` if the offset is covered by any range in the set, `false` otherwise.
    #[inline]
    pub fn contains_n(&self, offset: usize) -> bool {
        if let Some((_, last)) = self.0.range(..=offset).next_back() {
            return offset <= *last;
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
    pub fn contains(&self, range: &OffsetRange) -> bool {
        if let Some((_, last)) = self.0.range(..=range.start).next_back() {
            return range.last <= *last;
        }
        false
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
    pub fn insert_range(&mut self, rng: &OffsetRange) -> bool {
        // Position the cursor at the first position that might intersect with rng
        let mut cursor = self.0.upper_bound_mut(Bound::Included(&rng.start));
        if let Some(prev) = cursor.peek_prev().map(|(l, r)| OffsetRange::from((*l, *r)))
            && prev.intersects_or_adjacent(rng)
        {
            cursor.prev();
        }
        // Check if this is a no-op (containment)
        // We only need to check the element at the current cursor position.
        // If that element (the first one that might intersect) already contains
        // the new range, then the insertion is a no-op, so return false.
        if let Some(next) = cursor.peek_next().map(|(l, r)| OffsetRange::from((*l, *r)))
            && next.contains(rng)
        {
            return false;
        }
        // If it's not a no-op, perform the merge/insertion logic
        // Since we've excluded the fully contained case, any subsequent operations
        // will necessarily modify the set
        let mut merged_rng = *rng;
        // Continue looping as long as the next element exists and intersects with our range
        while cursor
            .peek_next()
            .map(|(l, r)| OffsetRange::from((*l, *r)))
            .is_some_and(|next| merged_rng.intersects_or_adjacent(&next))
        {
            // SAFETY: We've confirmed `peek_next()` returns `Some` in the loop condition,
            // so calling `remove_next()` will not panic. Using `unwrap_unchecked` is a
            // micro-optimization to avoid a redundant check.
            let rng_to_merge: OffsetRange =
                unsafe { cursor.remove_next().unwrap_unchecked().into() };
            // SAFETY: The loop condition `intersects_or_adjacent` guarantees that `merge`
            // will return `Some`, so this unwrap is safe.
            merged_rng = unsafe { merged_rng.merge(&rng_to_merge).unwrap_unchecked() };
        }
        unsafe {
            // safety:
            cursor
                .insert_after(merged_rng.start, merged_rng.last)
                .unwrap_unchecked()
        };
        true
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
    /// A new `OffsetRangeSet` containing the union of both sets.
    #[must_use]
    pub fn union_merge(&self, other: &Self) -> Self {
        let mut result = BTreeMap::new();
        let mut self_it = self.0.iter().peekable();
        let mut other_it = other.0.iter().peekable();

        // Store the current range being built that might still be expanded.
        let mut cur_merged: Option<OffsetRange> = None;

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
            let next_rng = OffsetRange::new(*next_rng_tuple.0, *next_rng_tuple.1);
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
        OffsetRangeSet(result)
    }

    /// Computes the union of two sets.
    ///
    /// This method chooses the most efficient algorithm based on the sizes of the sets.
    ///
    /// # Arguments
    ///
    /// * `other` - The other set to union with
    ///
    /// # Returns
    ///
    /// A new `OffsetRangeSet` containing the union of both sets.
    #[must_use]
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        if self.0.is_empty() {
            return other.clone();
        }
        if other.0.is_empty() {
            return self.clone();
        }
        let self_rngs_count = self.0.len();
        let other_rngs_count = other.0.len();
        let insert_cost_estimate = other_rngs_count * self_rngs_count.ilog2() as usize;
        let merge_cost_estimate = self_rngs_count + other_rngs_count;
        if insert_cost_estimate < merge_cost_estimate && other_rngs_count < self_rngs_count {
            let mut result = self.clone();
            for (&start, &last) in &other.0 {
                result.insert_range(&OffsetRange::new(start, last));
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
        let self_rngs_count = self.0.len();
        let other_rngs_count = other.0.len();
        let insert_cost_estimate = other_rngs_count * self_rngs_count.ilog2() as usize;
        let merge_cost_estimate = self_rngs_count + other_rngs_count;
        if insert_cost_estimate < merge_cost_estimate && other_rngs_count < self_rngs_count {
            for (&start, &last) in &other.0 {
                self.insert_range(&OffsetRange::new(start, last));
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
    /// A new `OffsetRangeSet` containing the difference.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        if self.0.is_empty() || other.0.is_empty() {
            return self.clone();
        }

        let mut result = OffsetRangeSet::new();
        let mut a_iter = self.0.iter();
        let mut b_iter = other.0.iter().peekable();

        // Get the first range from A
        let mut current_a = match a_iter.next() {
            Some((&start, &last)) => OffsetRange::new(start, last),
            None => return result, // A is empty
        };

        loop {
            // Look at B's next range
            match b_iter.peek() {
                // B still has ranges
                Some(&(&b_start, &b_last)) => {
                    let b_range = OffsetRange::new(b_start, b_last);

                    // If b_range is completely before current_a, skip this b_range
                    if b_range.last() < current_a.start() {
                        b_iter.next(); // Consume b_range
                        continue;
                    }

                    // If b_range is completely after current_a, current_a won't be trimmed further
                    // Complete processing of current_a, then try to get the next from A
                    if b_range.start() > current_a.last() {
                        result.insert_range(&current_a);
                        if let Some((&s, &l)) = a_iter.next() {
                            current_a = OffsetRange::new(s, l);
                            continue;
                        } else {
                            break;
                        }
                    }
                    // If b_range leaves a part before it in current_a
                    if b_range.start() > current_a.start() {
                        let prefix = OffsetRange::new(current_a.start(), b_range.start() - 1);
                        result.insert_range(&prefix);
                    }
                    // Update current_a's start, skipping the part covered by b_range
                    // If b_range.last() overflows, it means current_a is completely covered
                    if let Some(new_start) = b_range.last().checked_add(1) {
                        // If new_start is beyond current_a's range
                        if new_start > current_a.last() {
                            // current_a is completely processed, get the next one
                            current_a = match a_iter.next() {
                                Some((&s, &l)) => OffsetRange::new(s, l),
                                None => break, // A exhausted, end
                            };
                        } else {
                            // current_a still has remainder, update start and continue processing
                            current_a = OffsetRange::new(new_start, current_a.last());
                        }
                    } else {
                        // b_range.last() is usize::MAX, no remainder possible after current_a
                        current_a = match a_iter.next() {
                            Some((&s, &l)) => OffsetRange::new(s, l),
                            None => break, // A exhausted, end
                        };
                    }
                }
                // B is exhausted, all remaining ranges in A belong to the result
                None => {
                    result.insert_range(&current_a);
                    for (&start, &last) in a_iter {
                        result.insert_range(&OffsetRange::new(start, last));
                    }
                    break; // End main loop
                }
            }
        }
        result
    }

    /// Performs difference operation and assigns the result to self.
    ///
    /// Subtracts [other] from [self] and stores the result in [self].
    ///
    /// # Arguments
    ///
    /// * `other` - The set to subtract
    #[inline]
    pub fn difference_assign(&mut self, other: &OffsetRangeSet) {
        if self.0.is_empty() || other.0.is_empty() {
            return;
        }
        *self = self.difference(other);
    }
}

impl BitOrAssign<&OffsetRangeSet> for OffsetRangeSet {
    /// Performs the `|=` operation, equivalent to [union_assign].
    #[inline]
    fn bitor_assign(&mut self, rhs: &OffsetRangeSet) {
        self.union_assign(rhs);
    }
}

impl BitOr<&OffsetRangeSet> for OffsetRangeSet {
    type Output = OffsetRangeSet;

    /// Performs the `|` operation, equivalent to [union].
    #[inline]
    fn bitor(self, rhs: &OffsetRangeSet) -> Self::Output {
        self.union(rhs)
    }
}

impl SubAssign<&OffsetRangeSet> for OffsetRangeSet {
    /// Performs the `-=` operation, equivalent to [difference_assign].
    #[inline]
    fn sub_assign(&mut self, rhs: &OffsetRangeSet) {
        self.difference_assign(rhs);
    }
}

impl Sub<&OffsetRangeSet> for OffsetRangeSet {
    type Output = OffsetRangeSet;

    /// Performs the `-` operation, equivalent to [difference].
    #[inline]
    fn sub(self, rhs: &OffsetRangeSet) -> Self::Output {
        self.difference(rhs)
    }
}

/// An iterator that chunks an `OffsetRangeSet` into fixed-size blocks.
///
/// This iterator consumes an `OffsetRangeSet` and produces chunks of ranges
/// where each chunk has approximately the specified block size.
pub struct ChunkedMutIter<'a> {
    inner: &'a mut OffsetRangeSet,
    block_size: usize,
}

impl OffsetRangeSet {
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
    /// Panics in debug builds if `block_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sparse_ranges::{OffsetRange, OffsetRangeSet};
    /// let mut set = OffsetRangeSet::new();
    /// set.insert_range(&OffsetRange::new(0, 100));
    /// let mut chunks = set.into_chunks(10);
    /// // Process chunks...
    /// ```
    pub fn into_chunks(&mut self, block_size: usize) -> ChunkedMutIter<'_> {
        debug_assert!(block_size > 0, "block_size must be greater than 0");
        ChunkedMutIter {
            inner: self,
            block_size,
        }
    }
}

impl Iterator for ChunkedMutIter<'_> {
    type Item = Box<[OffsetRange]>;

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
        if self.inner.0.is_empty() {
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
                chunk_rngs.push(OffsetRange::new(start, last));
                // Update the chunk's remaining capacity
                remaining_size -= cur_rng_len;
            } else {
                // --- The current range is too large, only part of it fits ---
                // Calculate the end position that this chunk can accommodate from this range
                let chunk_last = start + remaining_size - 1;
                // Add that part to the chunk
                chunk_rngs.push(OffsetRange::new(start, chunk_last));
                // So we remove the old entry, then insert a new entry representing the remainder.
                let original_last = self.inner.0.pop_first().unwrap().1;
                self.inner.0.insert(chunk_last + 1, original_last);
                // The chunk is now full, force the loop to end
                remaining_size = 0;
            }
        }
        // If we successfully got any data from the BTreeMap,
        // return the constructed chunk, otherwise return None.
        if chunk_rngs.is_empty() {
            None
        } else {
            Some(chunk_rngs.into_boxed_slice())
        }
    }
}

impl<T: Into<OffsetRange>> FromIterator<T> for OffsetRangeSet {
    /// Creates an `OffsetRangeSet` from an iterator.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator of items that can be converted to `OffsetRange`
    ///
    /// # Returns
    ///
    /// A new `OffsetRangeSet` containing all the ranges from the iterator.
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = OffsetRangeSet::new();
        for item in iter {
            set.insert_range(&item.into());
        }
        set
    }
}

/// Error types that can occur when working with range sets.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// An error occurred when parsing a range header.
    #[cfg(feature = "http")]
    #[error(transparent)]
    Header(#[from] http_range_header::RangeUnsatisfiableError),
    #[error("invalid range unit")]
    Invalid,
    #[error("index overflow")]
    IndexOverflow,
    #[error("empty ranges")]
    Empty,
}

#[cfg(feature = "http")]
impl OffsetRangeSet {
    /// Parses an HTTP 'Range' header string relative to a total entity size.
    ///
    /// This function correctly handles all valid range formats, including:
    /// - `bytes=0-499` (absolute range)
    /// - `bytes=500-` (open-ended range)
    /// - `bytes=-100` (suffix range)
    ///
    /// It returns a `RangeUnsatisfiableError` if any calculated range is invalid
    /// with respect to the `total_size`, as per RFC 7233.
    pub fn parse_ranges_headers(
        header_content: &str,
        total_size: usize,
    ) -> Result<OffsetRangeSet, Error> {
        use http_range_header::{EndPosition, StartPosition};
        if total_size == 0 {
            // According to RFC 7233, a range header on a zero-length entity
            // is always unsatisfiable.
            return Err(Error::Empty);
        }

        let mut set = OffsetRangeSet::new();
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
                },
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
            set.insert_range(&OffsetRange::new(start, last_clamped));
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
    pub fn as_http_range_header(&self) -> Option<Box<str>> {
        if self.0.is_empty() {
            return None;
        }
        let parts: Vec<String> = self
            .0
            .iter()
            .map(|(&start, &last)| format!("{}-{}", start, last))
            .collect();
        Some(format!("bytes={}", parts.join(",")).into_boxed_str())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Error, OffsetRange, OffsetRangeSet};
    use std::{collections::BTreeMap, ops::RangeInclusive};

    fn btree_set(ranges: &[RangeInclusive<usize>]) -> BTreeMap<usize, usize> {
        ranges
            .iter()
            .map(|rng| (*rng.start(), *rng.end()))
            .collect()
    }

    #[test]
    fn test_offset_range_new() {
        let range = OffsetRange::new(5, 10);
        assert_eq!(range.start(), 5);
        assert_eq!(range.last(), 10);
        assert_eq!(range.len(), 6);
    }

    #[test]
    fn test_offset_range_contains_n() {
        let range = OffsetRange::new(5, 10);
        assert!(range.contains_n(5));
        assert!(range.contains_n(7));
        assert!(range.contains_n(10));
        assert!(!range.contains_n(4));
        assert!(!range.contains_n(11));
    }

    #[test]
    fn test_offset_range_contains() {
        let range = OffsetRange::new(5, 10);
        assert!(range.contains(&OffsetRange::new(5, 10))); // Same range
        assert!(range.contains(&OffsetRange::new(6, 9))); // Strictly inside
        assert!(!range.contains(&OffsetRange::new(4, 10))); // Extends before
        assert!(!range.contains(&OffsetRange::new(5, 11))); // Extends after
        assert!(!range.contains(&OffsetRange::new(3, 4))); // Completely before
        assert!(!range.contains(&OffsetRange::new(11, 12))); // Completely after
    }

    #[test]
    fn test_offset_range_intersects() {
        let range = OffsetRange::new(5, 10);
        assert!(range.intersects(&OffsetRange::new(5, 10))); // Same range
        assert!(range.intersects(&OffsetRange::new(3, 7))); // Overlaps start
        assert!(range.intersects(&OffsetRange::new(8, 12))); // Overlaps end
        assert!(range.intersects(&OffsetRange::new(3, 12))); // Contains range
        assert!(range.intersects(&OffsetRange::new(6, 9))); // Contained in range
        assert!(!range.intersects(&OffsetRange::new(2, 4))); // Completely before
        assert!(!range.intersects(&OffsetRange::new(11, 15))); // Completely after
    }

    #[test]
    fn test_offset_range_intersects_or_adjacent() {
        let range = OffsetRange::new(5, 10);
        assert!(range.intersects_or_adjacent(&OffsetRange::new(5, 10))); // Same range
        assert!(range.intersects_or_adjacent(&OffsetRange::new(3, 7))); // Overlaps start
        assert!(range.intersects_or_adjacent(&OffsetRange::new(8, 12))); // Overlaps end
        assert!(range.intersects_or_adjacent(&OffsetRange::new(3, 12))); // Contains range
        assert!(range.intersects_or_adjacent(&OffsetRange::new(6, 9))); // Contained in range
        assert!(range.intersects_or_adjacent(&OffsetRange::new(2, 4))); // Adjacent before
        assert!(range.intersects_or_adjacent(&OffsetRange::new(11, 15))); // Adjacent after
        assert!(!range.intersects_or_adjacent(&OffsetRange::new(1, 3))); // Separated before
        assert!(!range.intersects_or_adjacent(&OffsetRange::new(12, 15))); // Separated after
    }

    #[test]
    fn test_offset_range_merge() {
        let range = OffsetRange::new(5, 10);

        // Merge with overlapping range
        assert_eq!(
            range.merge(&OffsetRange::new(8, 15)),
            Some(OffsetRange::new(5, 15))
        );

        // Merge with adjacent range (after)
        assert_eq!(
            range.merge(&OffsetRange::new(11, 15)),
            Some(OffsetRange::new(5, 15))
        );

        // Merge with adjacent range (before)
        assert_eq!(
            range.merge(&OffsetRange::new(1, 4)),
            Some(OffsetRange::new(1, 10))
        );

        // Cannot merge with separated range
        assert_eq!(range.merge(&OffsetRange::new(12, 15)), None);
    }

    #[test]
    fn test_offset_range_try_from_range() {
        // Valid range
        let range: Result<OffsetRange, _> = (&(5..11)).try_into();
        assert_eq!(range, Ok(OffsetRange::new(5, 10)));

        // Test with a range that would cause underflow when converting to inclusive (end == 0)
        let range: Result<OffsetRange, _> = (&(0..0)).try_into();
        assert!(range.is_err());
        assert!(matches!(range.unwrap_err(), Error::IndexOverflow));
    }

    #[test]
    fn test_offset_range_from_range_inclusive() {
        let std_range = 5..=10;
        let range = OffsetRange::from(&std_range);
        assert_eq!(range, OffsetRange::new(5, 10));
    }

    #[test]
    fn test_offset_range_from_tuple() {
        let range = OffsetRange::from((5, 10));
        assert_eq!(range, OffsetRange::new(5, 10));
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
    fn test_offset_range_set_new() {
        let set = OffsetRangeSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_offset_range_set_len() {
        let mut set = OffsetRangeSet::new();
        assert_eq!(set.len(), 0);

        set.insert_range(&OffsetRange::new(0, 5)); // 6 elements
        assert_eq!(set.len(), 6);

        set.insert_range(&OffsetRange::new(10, 12)); // 3 more elements
        assert_eq!(set.len(), 9);

        set.insert_range(&OffsetRange::new(3, 11)); // Merge all into one range (0-12) = 13 elements
        assert_eq!(set.len(), 13);
    }

    #[test]
    fn test_offset_range_set_contains_n() {
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(5, 10));
        set.insert_range(&OffsetRange::new(15, 20));

        // Test elements in ranges
        assert!(set.contains_n(5));
        assert!(set.contains_n(10));
        assert!(set.contains_n(15));
        assert!(set.contains_n(20));

        // Test elements between ranges
        assert!(!set.contains_n(11));
        assert!(!set.contains_n(14));

        // Test elements outside ranges
        assert!(!set.contains_n(4));
        assert!(!set.contains_n(21));
    }

    #[test]
    fn test_offset_range_set_contains() {
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(5, 10));
        set.insert_range(&OffsetRange::new(15, 20));

        // Test ranges fully contained
        assert!(set.contains(&OffsetRange::new(5, 10)));
        assert!(set.contains(&OffsetRange::new(15, 20)));
        assert!(set.contains(&OffsetRange::new(6, 9)));
        assert!(set.contains(&OffsetRange::new(16, 19)));

        // Test ranges partially contained
        assert!(!set.contains(&OffsetRange::new(4, 6)));
        assert!(!set.contains(&OffsetRange::new(9, 11)));
        assert!(!set.contains(&OffsetRange::new(14, 16)));
        assert!(!set.contains(&OffsetRange::new(19, 21)));

        // Test ranges not contained at all
        assert!(!set.contains(&OffsetRange::new(11, 14)));
        assert!(!set.contains(&OffsetRange::new(2, 4)));
        assert!(!set.contains(&OffsetRange::new(21, 25)));
    }

    #[test]
    #[cfg(feature = "http")]
    fn test_parse_ranges_headers_valid() {
        // Test absolute range
        let set = OffsetRangeSet::parse_ranges_headers("bytes=0-499", 1000).unwrap();
        assert_eq!(set.len(), 500);
        assert!(set.contains_n(0));
        assert!(set.contains_n(499));
        assert!(!set.contains_n(500));

        // Test open-ended range
        let set = OffsetRangeSet::parse_ranges_headers("bytes=500-", 1000).unwrap();
        assert_eq!(set.len(), 500);
        assert!(set.contains_n(500));
        assert!(set.contains_n(999));
        assert!(!set.contains_n(499));

        // Test suffix range
        let set = OffsetRangeSet::parse_ranges_headers("bytes=-100", 1000).unwrap();
        assert_eq!(set.len(), 100);
        assert!(set.contains_n(900));
        assert!(set.contains_n(999));
        assert!(!set.contains_n(899));

        // Test multiple ranges
        let set = OffsetRangeSet::parse_ranges_headers("bytes=0-5,10-15", 1000).unwrap();
        assert_eq!(set.len(), 12);
        assert!(set.contains_n(0));
        assert!(set.contains_n(5));
        assert!(!set.contains_n(6));
        assert!(!set.contains_n(9));
        assert!(set.contains_n(10));
        assert!(set.contains_n(15));
        assert!(!set.contains_n(16));
    }

    #[test]
    #[cfg(feature = "http")]
    fn test_parse_ranges_headers_invalid() {
        // Test invalid range (start > end)
        let result = OffsetRangeSet::parse_ranges_headers("bytes=500-400", 1000);
        assert!(matches!(result, Err(Error::Invalid)));
        
        // Test invalid range (start >= total_size)
        let result = OffsetRangeSet::parse_ranges_headers("bytes=1000-", 1000);
        assert!(matches!(result, Err(Error::Invalid)));
        
        // Test invalid range (start > total_size)
        let result = OffsetRangeSet::parse_ranges_headers("bytes=1001-2000", 1000);
        assert!(matches!(result, Err(Error::Invalid)));
        
        // Test invalid format (FromLast with Index) - this is handled by the http_range_header crate
        // and would return a Header error, not our Invalid error
        let result = OffsetRangeSet::parse_ranges_headers("bytes=-500-600", 1000);
        assert!(result.is_err());
        
        // Test empty range (-0) - this would also be handled by the http_range_header crate
        // and would return a Header error, not our Empty error
        let result = OffsetRangeSet::parse_ranges_headers("bytes=-0", 1000);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "http")]
    fn test_parse_ranges_headers_empty_entity() {
        // Test any range on zero-length entity
        let result = OffsetRangeSet::parse_ranges_headers("bytes=0-499", 0);
        assert!(matches!(result, Err(Error::Empty)));

        let result = OffsetRangeSet::parse_ranges_headers("bytes=500-", 0);
        assert!(matches!(result, Err(Error::Empty)));

        let result = OffsetRangeSet::parse_ranges_headers("bytes=-100", 0);
        assert!(matches!(result, Err(Error::Empty)));
    }

    #[test]
    #[cfg(feature = "http")]
    fn test_parse_ranges_headers_clamping() {
        // Test range clamping
        let set = OffsetRangeSet::parse_ranges_headers("bytes=0-1000", 500).unwrap();
        assert_eq!(set.len(), 500);
        assert!(set.contains_n(0));
        assert!(set.contains_n(499));
        assert!(!set.contains_n(500));
    }

    #[test]
    #[cfg(feature = "http")]
    fn test_as_http_range_header() {
        let mut set = OffsetRangeSet::new();

        // Test empty set
        assert_eq!(set.as_http_range_header(), None);

        // Test single range
        set.insert_range(&OffsetRange::new(0, 5));
        assert_eq!(set.as_http_range_header(), Some("bytes=0-5".into()));

        // Test multiple ranges
        set.insert_range(&OffsetRange::new(10, 15));
        assert_eq!(set.as_http_range_header(), Some("bytes=0-5,10-15".into()));
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
    fn test_union_edge_cases() {
        // Test union with self
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(5, 10));
        let union = set.union(&set);
        assert_eq!(union.len(), 6);
        assert!(union.contains_n(5));
        assert!(union.contains_n(10));

        // Test union with overlapping sets
        let mut set1 = OffsetRangeSet::new();
        set1.insert_range(&OffsetRange::new(5, 15));

        let mut set2 = OffsetRangeSet::new();
        set2.insert_range(&OffsetRange::new(10, 20));

        let union = set1.union(&set2);
        assert_eq!(union.len(), 16); // 5-20
        assert!(union.contains_n(5));
        assert!(union.contains_n(20));
        assert!(!union.contains_n(4));
        assert!(!union.contains_n(21));
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

    #[test]
    fn test_difference_edge_cases() {
        // Test difference with self
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(5, 10));
        let difference = set.difference(&set);
        assert!(difference.is_empty());
        assert_eq!(difference.len(), 0);

        // Test difference with non-overlapping sets
        let mut set1 = OffsetRangeSet::new();
        set1.insert_range(&OffsetRange::new(5, 10));

        let mut set2 = OffsetRangeSet::new();
        set2.insert_range(&OffsetRange::new(15, 20));

        let difference = set1.difference(&set2);
        assert_eq!(difference.len(), 6);
        assert!(difference.contains_n(5));
        assert!(difference.contains_n(10));
        assert!(!difference.contains_n(11));
    }

    #[test]
    fn test_chunks_gathers_multiple_discrete_ranges() {
        let mut set = range_set(&[0..=10, 90..=100]); // len=11, len=11

        // Block size 30 应该足以容纳这两个区间
        let mut chunks_iter = set.into_chunks(30);

        // 第一个块应该包含两个区间
        let first_chunk = chunks_iter.next().unwrap();
        assert_eq!(first_chunk.len(), 2);
        assert_eq!(first_chunk[0], OffsetRange::new(0, 10));
        assert_eq!(first_chunk[1], OffsetRange::new(90, 100));

        // 之后应该没有更多块了
        assert!(chunks_iter.next().is_none());
        assert!(set.0.is_empty());
    }

    #[test]
    fn test_chunks_gathers_and_splits() {
        let mut set = range_set(&[
            0..=5,     // len = 6
            10..=15,   // len = 6
            100..=150, // len = 51
        ]);

        // Block size 20
        let mut chunks_iter = set.into_chunks(20);

        // 第一个块：应该包含 0..=5 和 10..=15，总长度 12。
        // 剩余空间 8。然后会取 100..=150 的前8个元素。
        let chunk1 = chunks_iter.next().unwrap();
        assert_eq!(chunk1.len(), 3);
        assert_eq!(chunk1[0], OffsetRange::new(0, 5));
        assert_eq!(chunk1[1], OffsetRange::new(10, 15));
        assert_eq!(chunk1[2], OffsetRange::new(100, 107)); // 6 + 6 + 8 = 20

        // 第二个块：处理 100..=150 的剩余部分，取 20 个
        let chunk2 = chunks_iter.next().unwrap();
        assert_eq!(chunk2.len(), 1);
        assert_eq!(chunk2[0], OffsetRange::new(108, 127));

        // 第三个块：继续处理
        let chunk3 = chunks_iter.next().unwrap();
        assert_eq!(chunk3.len(), 1);
        assert_eq!(chunk3[0], OffsetRange::new(128, 147));

        // 第四个块：最后剩余的部分
        let chunk4 = chunks_iter.next().unwrap();
        assert_eq!(chunk4.len(), 1);
        assert_eq!(chunk4[0], OffsetRange::new(148, 150));

        // 结束
        assert!(chunks_iter.next().is_none());
        assert!(set.0.is_empty());
    }

    #[test]
    fn test_chunks_empty_set() {
        let mut set = OffsetRangeSet::new();
        let mut chunks = set.into_chunks(10);
        assert!(chunks.next().is_none());
    }

    #[test]
    #[should_panic(expected = "block_size must be greater than 0")]
    fn test_chunks_zero_block_size() {
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(0, 5));
        let mut chunks = set.into_chunks(0);
        // This should panic due to the debug_assert! in into_chunks
        let _ = chunks.next();
    }

    #[test]
    fn test_chunks_exact_block_size() {
        let mut set = OffsetRangeSet::new();
        set.insert_range(&OffsetRange::new(0, 9)); // Exactly 10 elements

        let mut chunks = set.into_chunks(10);
        let chunk = chunks.next().unwrap();
        assert_eq!(chunk.len(), 1);
        assert_eq!(chunk[0], OffsetRange::new(0, 9));
        assert!(chunks.next().is_none());
        assert!(set.is_empty());
    }
}
