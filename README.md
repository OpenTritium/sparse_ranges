# sparse_ranges

Efficient sparse range set operations for Rust, particularly designed for HTTP range requests and file offset management.

## Overview

`sparse_ranges` is a Rust library that efficiently represents and manipulates sparse ranges - non-contiguous sets of integer ranges. It's especially useful for handling HTTP range requests and managing file offsets where you need to work with multiple, potentially overlapping or adjacent byte ranges.

The library provides:
- Efficient storage of non-overlapping ranges
- Automatic merging of overlapping or adjacent ranges
- Operations like union, difference, and intersection
- Parsing and generation of HTTP Range headers (with the `http` feature)
- Chunking functionality for processing large ranges in smaller blocks

## Requirements

This library requires a nightly Rust compiler due to its use of the experimental `btree_cursors` feature from the 2024 edition. To use this library, ensure you have a recent nightly version of Rust installed.

The `#![feature(btree_cursors)]` attribute is used to enable experimental APIs for working with BTreeMap cursors, which provides efficient range manipulation capabilities.

## Usage Examples

### Basic Operations

```rust
use sparse_ranges::{Range, RangeSet};

let mut ranges = RangeSet::new();
ranges.insert_range(&Range::new(0, 10));
ranges.insert_range(&Range::new(20, 30));
```

### Set Operations with Operators

```rust
use sparse_ranges::{Range, RangeSet};
let mut set1 = RangeSet::new();
set1.insert_range(&Range::new(0, 10));
let mut set2 = RangeSet::new();
set2.insert_range(&Range::new(5, 15));

// Union with | operator
let set3 = &set1 | &set2;

// Difference with - operator
let diff = &set1 - &set2;

// Union assignment with |= operator
set1 |= &set2;

// Difference assignment with -= operator
set1 -= &set2;
```

### Freeze

Create an immutable snapshot of a range set that converts the internal BTreeMap structure to a more efficient array format:

```rust
use sparse_ranges::{Range, RangeSet};

let mut ranges = RangeSet::new();
ranges.insert_range(&Range::new(0, 10));
ranges.insert_range(&Range::new(20, 30));

// Create an immutable snapshot with array-based storage
let frozen = ranges.freeze();

// FrozenRangeSet uses array storage which has better cache locality
// and provides efficient iteration while preserving all query operations
assert_eq!(frozen.len(), 22); // Total elements in all ranges
assert_eq!(frozen.ranges_count(), 2); // Number of separate ranges

// You can still iterate over the ranges
for range in frozen.iter() {
    println!("Range: {:?}", range);
}
```

The `freeze` operation creates a `FrozenRangeSet` which:
- Converts the internal BTreeMap structure to an array for better cache locality
- Provides immutable but efficient access to range data
- Maintains all query capabilities (contains, contains_n, etc.)
- Allows efficient iteration over the ranges

### Chunking

Process large ranges in smaller blocks:

```rust
use sparse_ranges::{Range, RangeSet};

let mut ranges = RangeSet::new();
ranges.insert_range(&Range::new(0, 1000));

// Process ranges in chunks of ~100 elements each
let mut chunks = ranges.into_chunks(100);
while let Some(chunk) = chunks.next() {
    // Each chunk is a FrozenRangeSet containing ranges
    // with approximately 100 elements total
    process_chunk(chunk);
}
```