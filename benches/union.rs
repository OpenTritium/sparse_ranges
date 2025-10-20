use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use range_set_blaze::RangeSetBlaze; // 假设 RangeSetBlaze 实现了 | (BitOr) 操作符用于求并集
use sparse_ranges::{OffsetRange, OffsetRangeSet};

// --- 常量定义 (与您的文件保持一致) ---
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

fn union_benchmark(c: &mut Criterion) {
    // --- 测试数据准备 ---
    // 我们需要为 union 操作准备两个集合。
    // 我们将总共 SET_SIZE 个范围平分到两个集合中。
    let all_random_ranges = generate_random_ranges(SET_SIZE);
    let (set1_random_ranges, set2_random_ranges) = all_random_ranges.split_at(SET_SIZE / 2);

    let all_sequential_ranges: Vec<_> = (0..SET_SIZE).map(|i| (i * 10, i * 10 + 5)).collect();
    let (set1_seq_ranges, set2_seq_ranges) = all_sequential_ranges.split_at(SET_SIZE / 2);

    // --- Benchmark 分组 ---
    let mut group = c.benchmark_group("Union Performance");

    // --- 场景 1: 顺序范围的并集 (Sequential Union) ---
    // 预先构建好用于测试的集合，这部分开销不计入 benchmark
    let mut set1_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set1_seq_ranges {
        set1_seq_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set2_seq_offset = OffsetRangeSet::new();
    for &(start, end) in set2_seq_ranges {
        set2_seq_offset.insert_range(&OffsetRange::new(start, end));
    }

    let mut set1_seq_blaze = RangeSetBlaze::new();
    for &(start, end) in set1_seq_ranges {
        set1_seq_blaze.ranges_insert(start..=end);
    }
    let mut set2_seq_blaze = RangeSetBlaze::new();
    for &(start, end) in set2_seq_ranges {
        set2_seq_blaze.ranges_insert(start..=end);
    }

    group.bench_function("OffsetRangeSet - Sequential Union", |b| {
        b.iter(|| {
            // 只测试 union 操作本身
            let _ = black_box(&set1_seq_offset).union(black_box(&set2_seq_offset));
        })
    });

    group.bench_function("RangeSetBlaze - Sequential Union", |b| {
        b.iter(|| {
            // RangeSetBlaze 通常使用 | 操作符来求并集
            let _ = black_box(&set1_seq_blaze) | black_box(&set2_seq_blaze);
        })
    });

    // --- 场景 2: 随机范围的并集 (Random Union) ---
    // 预先构建好用于测试的集合
    let mut set1_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set1_random_ranges {
        set1_rand_offset.insert_range(&OffsetRange::new(start, end));
    }
    let mut set2_rand_offset = OffsetRangeSet::new();
    for &(start, end) in set2_random_ranges {
        set2_rand_offset.insert_range(&OffsetRange::new(start, end));
    }

    let mut set1_rand_blaze = RangeSetBlaze::new();
    for &(start, end) in set1_random_ranges {
        set1_rand_blaze.ranges_insert(start..=end);
    }
    let mut set2_rand_blaze = RangeSetBlaze::new();
    for &(start, end) in set2_random_ranges {
        set2_rand_blaze.ranges_insert(start..=end);
    }

    group.bench_function("OffsetRangeSet - Random Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_offset).union(black_box(&set2_rand_offset));
        })
    });

    group.bench_function("RangeSetBlaze - Random Union", |b| {
        b.iter(|| {
            let _ = black_box(&set1_rand_blaze) | black_box(&set2_rand_blaze);
        })
    });

    group.finish();
}

// 将两个 benchmark 函数都注册到 criterion
criterion_group!(benches, union_benchmark);
criterion_main!(benches);
