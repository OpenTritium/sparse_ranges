use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use range_set_blaze::RangeSetBlaze;
use sparse_ranges::{Range, RangeSet};

// --- 常量定义 ---
const SET_SIZE: usize = 1000;
const RANGE_MAX: usize = 10_000;

/// 生成一系列用于测试的随机范围
fn generate_random_ranges(count: usize) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(42);
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

fn intersection_benchmark(c: &mut Criterion) {
    // --- 测试数据准备 ---
    let all_random_ranges = generate_random_ranges(SET_SIZE);
    let (set1_random_ranges, set2_random_ranges) = all_random_ranges.split_at(SET_SIZE / 2);

    let all_sequential_ranges: Vec<_> = (0..SET_SIZE).map(|i| (i * 10, i * 10 + 5)).collect();
    let (set1_seq_ranges, set2_seq_ranges) = all_sequential_ranges.split_at(SET_SIZE / 2);

    // --- Benchmark 分组 ---
    let mut group = c.benchmark_group("Intersection Performance");

    // --- 场景 1: 顺序范围的交集 (Sequential Intersection) ---
    let mut set1_seq_offset = RangeSet::new();
    for &(start, end) in set1_seq_ranges {
        set1_seq_offset.insert_range(&Range::new(start, end));
    }
    let mut set2_seq_offset = RangeSet::new();
    for &(start, end) in set2_seq_ranges {
        set2_seq_offset.insert_range(&Range::new(start, end));
    }

    let mut set1_seq_blaze = RangeSetBlaze::new();
    for &(start, end) in set1_seq_ranges {
        set1_seq_blaze.ranges_insert(start..=end);
    }
    let mut set2_seq_blaze = RangeSetBlaze::new();
    for &(start, end) in set2_seq_ranges {
        set2_seq_blaze.ranges_insert(start..=end);
    }

    group.bench_function("OffsetRangeSet - Sequential Intersection (insert)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_offset).intersection_insert(black_box(&set2_seq_offset));
        })
    });

    group.bench_function("OffsetRangeSet - Sequential Intersection (merge)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_offset).intersection_merge(black_box(&set2_seq_offset));
        })
    });

    group.bench_function("OffsetRangeSet - Sequential Intersection (default)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_offset).intersection(black_box(&set2_seq_offset));
        })
    });

    group.bench_function("RangeSetBlaze - Sequential Intersection", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_blaze) & black_box(&set2_seq_blaze);
        })
    });

    // --- 场景 2: 随机范围的交集 (Random Intersection) ---
    let mut set1_rand_offset = RangeSet::new();
    for &(start, end) in set1_random_ranges {
        set1_rand_offset.insert_range(&Range::new(start, end));
    }
    let mut set2_rand_offset = RangeSet::new();
    for &(start, end) in set2_random_ranges {
        set2_rand_offset.insert_range(&Range::new(start, end));
    }

    let mut set1_rand_blaze = RangeSetBlaze::new();
    for &(start, end) in set1_random_ranges {
        set1_rand_blaze.ranges_insert(start..=end);
    }
    let mut set2_rand_blaze = RangeSetBlaze::new();
    for &(start, end) in set2_random_ranges {
        set2_rand_blaze.ranges_insert(start..=end);
    }

    group.bench_function("OffsetRangeSet - Random Intersection (insert)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_offset).intersection_insert(black_box(&set2_rand_offset));
        })
    });

    group.bench_function("OffsetRangeSet - Random Intersection (merge)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_offset).intersection_merge(black_box(&set2_rand_offset));
        })
    });

    group.bench_function("OffsetRangeSet - Random Intersection (default)", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_offset).intersection(black_box(&set2_rand_offset));
        })
    });

    group.bench_function("RangeSetBlaze - Random Intersection", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_blaze) & black_box(&set2_rand_blaze);
        })
    });

    // --- 场景 3: 部分重叠的交集 (Partial Overlap) ---
    // 创建两个有明显重叠部分的集合
    let mut set1_overlap_offset = RangeSet::new();
    let mut set2_overlap_offset = RangeSet::new();
    let mut set1_overlap_blaze = RangeSetBlaze::new();
    let mut set2_overlap_blaze = RangeSetBlaze::new();

    // set1: 0-5000 范围内的数据
    for i in 0..500 {
        set1_overlap_offset.insert_range(&Range::new(i * 10, i * 10 + 5));
        set1_overlap_blaze.ranges_insert(i * 10..=i * 10 + 5);
    }

    // set2: 2500-7500 范围内的数据 (与 set1 有 2500-5000 的重叠)
    for i in 250..750 {
        set2_overlap_offset.insert_range(&Range::new(i * 10, i * 10 + 5));
        set2_overlap_blaze.ranges_insert(i * 10..=i * 10 + 5);
    }

    group.bench_function("OffsetRangeSet - Partial Overlap Intersection", |b| {
        b.iter(|| {
            let _ = black_box(&set1_overlap_offset).intersection(black_box(&set2_overlap_offset));
        })
    });

    group.bench_function("RangeSetBlaze - Partial Overlap Intersection", |b| {
        b.iter(|| {
            let _ = black_box(&set1_overlap_blaze) & black_box(&set2_overlap_blaze);
        })
    });

    group.finish();
}

criterion_group!(benches, intersection_benchmark);
criterion_main!(benches);
