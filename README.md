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