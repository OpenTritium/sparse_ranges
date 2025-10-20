use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use range_set_blaze::RangeSetBlaze;
use sparse_ranges::{OffsetRange, OffsetRangeSet};

const SET_SIZE: usize = 1000;
const RANGE_MAX: usize = 10_000;

/// 生成一系列用于测试的随机范围
fn generate_random_ranges(count: usize) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(42); // 使用固定的种子以保证每次测试数据一致
    let mut ranges = Vec::with_capacity(count);
    for _ in 0..count {
        let a = rng.random_range(0..RANGE_MAX);
        let b = rng.random_range(0..RANGE_MAX);
        if a < b {
            ranges.push((a, b));
        } else {
            ranges.push((b, a));
        }
    }
    ranges
}

/// Benchmark 函数
fn insertion_benchmark(c: &mut Criterion) {
    // --- 测试数据准备 ---
    let random_ranges: Vec<_> = generate_random_ranges(SET_SIZE);

    let sequential_ranges: Vec<_> = (0..SET_SIZE).map(|i| (i * 10, i * 10 + 5)).collect();

    let mut reverse_ranges = sequential_ranges.clone();
    reverse_ranges.reverse();

    // --- Benchmark 分组 ---
    let mut group = c.benchmark_group("Insertion Performance");

    // --- 场景 1: 顺序插入 (Sequential) ---
    group.bench_function("OffsetRangeSet - Sequential", |b| {
        b.iter(|| {
            let mut set = OffsetRangeSet::new();
            for &(start, end) in black_box(&sequential_ranges) {
                set.insert_range(&OffsetRange::new(start, end));
            }
        })
    });

    group.bench_function("RangeSetBlaze - Sequential", |b| {
        b.iter(|| {
            let mut set = RangeSetBlaze::new();
            for &(start, end) in black_box(&sequential_ranges) {
                set.ranges_insert(start..=end);
            }
        })
    });

    // --- 场景 2: 反向插入 (Reverse) ---
    group.bench_function("OffsetRangeSet - Reverse", |b| {
        b.iter(|| {
            let mut set = OffsetRangeSet::new();
            for &(start, end) in black_box(&reverse_ranges) {
                set.insert_range(&OffsetRange::new(start, end));
            }
        })
    });

    group.bench_function("RangeSetBlaze - Reverse", |b| {
        b.iter(|| {
            let mut set = RangeSetBlaze::new();
            for &(start, end) in black_box(&reverse_ranges) {
                set.ranges_insert(start..=end);
            }
        })
    });

    // --- 场景 3: 随机插入 (Random) ---
    group.bench_function("OffsetRangeSet - Random", |b| {
        b.iter(|| {
            let mut set = OffsetRangeSet::new();
            for &(start, end) in black_box(&random_ranges) {
                set.insert_range(&OffsetRange::new(start, end));
            }
        })
    });

    group.bench_function("RangeSetBlaze - Random", |b| {
        b.iter(|| {
            let mut set = RangeSetBlaze::new();
            for &(start, end) in black_box(&random_ranges) {
                set.ranges_insert(start..=end);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, insertion_benchmark);
criterion_main!(benches);
