use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use range_set_blaze::RangeSetBlaze;
use sparse_ranges::{OffsetRange, OffsetRangeSet};

// --- 常量定义 (保持一致) ---
const SET_SIZE: usize = 1000;
const RANGE_MAX: usize = 10_000;

// --- 辅助函数 (复用) ---
fn generate_random_ranges(count: usize) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(42); // 固定种子以保证可复现
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

// --- Union 性能测试 (保留您的函数) ---
fn union_benchmark(c: &mut Criterion) {
    let all_random_ranges = generate_random_ranges(SET_SIZE);
    let (set1_random_ranges, set2_random_ranges) = all_random_ranges.split_at(SET_SIZE / 2);

    let all_sequential_ranges: Vec<_> = (0..SET_SIZE).map(|i| (i * 10, i * 10 + 5)).collect();
    let (set1_seq_ranges, set2_seq_ranges) = all_sequential_ranges.split_at(SET_SIZE / 2);

    let mut group = c.benchmark_group("Union Performance");

    // --- 场景 1: 顺序范围的并集 ---
    let mut set1_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set1_seq_ranges {
        set1_seq_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set2_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set2_seq_ranges {
        set2_seq_offset.insert_range(&OffsetRange::new(start, end));
    }
    let set1_seq_blaze = RangeSetBlaze::from_iter(set1_seq_ranges.iter().map(|(s, e)| *s..=*e));
    let set2_seq_blaze = RangeSetBlaze::from_iter(set2_seq_ranges.iter().map(|(s, e)| *s..=*e));

    group.bench_function("OffsetRangeSet - Sequential Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_offset).union_merge(black_box(&set2_seq_offset));
        })
    });
    group.bench_function("RangeSetBlaze - Sequential Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_seq_blaze) | black_box(&set2_seq_blaze);
        })
    });

    // --- 场景 2: 随机范围的并集 ---
    let mut set1_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set1_random_ranges {
        set1_rand_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set2_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set2_random_ranges {
        set2_rand_offset.insert_range(&OffsetRange::new(start, end));
    }
    let set1_rand_blaze = RangeSetBlaze::from_iter(set1_random_ranges.iter().map(|(s, e)| *s..=*e));
    let set2_rand_blaze = RangeSetBlaze::from_iter(set2_random_ranges.iter().map(|(s, e)| *s..=*e));

    group.bench_function("OffsetRangeSet - Random Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_offset).union_merge(black_box(&set2_rand_offset));
        })
    });
    group.bench_function("RangeSetBlaze - Random Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_blaze) | black_box(&set2_rand_blaze);
        })
    });

    group.finish();
}

// --- 以下是为 Difference 操作编写的全新性能测试 ---
fn difference_benchmark(c: &mut Criterion) {
    // --- 测试数据准备 (与 union 类似) ---
    // set_a - set_b
    let all_random_ranges = generate_random_ranges(SET_SIZE);
    let (set_a_random_ranges, set_b_random_ranges) = all_random_ranges.split_at(SET_SIZE / 2);

    let all_sequential_ranges: Vec<_> = (0..SET_SIZE).map(|i| (i * 10, i * 10 + 5)).collect();
    let (set_a_seq_ranges, set_b_seq_ranges) = all_sequential_ranges.split_at(SET_SIZE / 2);

    // --- Benchmark 分组 ---
    let mut group = c.benchmark_group("Difference Performance");

    // --- 场景 1: 顺序范围的差集 (Sequential Difference) ---
    // 预先构建好用于测试的集合
    let mut set_a_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set_a_seq_ranges {
        set_a_seq_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set_b_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set_b_seq_ranges {
        set_b_seq_offset.insert_range(&OffsetRange::new(start, end));
    }
    let set_a_seq_blaze = RangeSetBlaze::from_iter(set_a_seq_ranges.iter().map(|(s, e)| *s..=*e));
    let set_b_seq_blaze = RangeSetBlaze::from_iter(set_b_seq_ranges.iter().map(|(s, e)| *s..=*e));

    group.bench_function("OffsetRangeSet - Sequential Difference", |b| {
        b.iter(|| {
            // 只测试 difference 操作本身
            let _ = black_box(&set_a_seq_offset).difference(black_box(&set_b_seq_offset));
        })
    });
    group.bench_function("RangeSetBlaze - Sequential Difference", |b| {
        b.iter(|| {
            // RangeSetBlaze 使用 - 操作符来求差集
            let _ = black_box(&set_a_seq_blaze) - black_box(&set_b_seq_blaze);
        })
    });

    // --- 场景 2: 随机范围的差集 (Random Difference) ---
    // 预先构建好用于测试的集合
    let mut set_a_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set_a_random_ranges {
        set_a_rand_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set_b_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set_b_random_ranges {
        set_b_rand_offset.insert_range(&OffsetRange::new(start, end));
    }
    let set_a_rand_blaze =
        RangeSetBlaze::from_iter(set_a_random_ranges.iter().map(|(s, e)| *s..=*e));
    let set_b_rand_blaze =
        RangeSetBlaze::from_iter(set_b_random_ranges.iter().map(|(s, e)| *s..=*e));

    group.bench_function("OffsetRangeSet - Random Difference", |b| {
        b.iter(|| {
            let _ = black_box(&set_a_rand_offset).difference(black_box(&set_b_rand_offset));
        })
    });
    group.bench_function("RangeSetBlaze - Random Difference", |b| {
        b.iter(|| {
            let _ = black_box(&set_a_rand_blaze) - black_box(&set_b_rand_blaze);
        })
    });

    group.finish();
}

// 将两个 benchmark 函数都注册到 criterion
criterion_group!(benches, difference_benchmark);
criterion_main!(benches);
