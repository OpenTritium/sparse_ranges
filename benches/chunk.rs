use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use sparse_ranges::{OffsetRange, OffsetRangeSet};

/// 场景 1: 测试处理大量离散小区间的性能。
///
/// 这种情况下，迭代器需要不断地从 BTreeMap 中弹出小区间，
/// 并将它们收集到同一个块中。
fn bench_chunks_many_small_ranges(c: &mut Criterion) {
    let mut group = c.benchmark_group("Many Small Ranges");

    // 创建一个包含 1000 个小区间（长度为 6）的初始集合
    let mut initial_set = OffsetRangeSet::new();
    for i in 0..1000 {
        initial_set.insert_range(&OffsetRange::new(i * 10, i * 10 + 5));
    }

    // 设置一个足够大的块大小，以确保每个块都能收集多个小区间
    let block_size = 1024;

    group.bench_function("gather_small_ranges", |b| {
        // b.iter 会多次运行闭包内的代码以收集性能数据
        b.iter(|| {
            // 在每次迭代中，我们必须克隆初始集合，
            // 因为 into_chunks() 是消耗性的。
            let mut set = initial_set.clone();

            // 使用 black_box 防止编译器优化掉对结果的计算，
            // collect() 会驱动整个迭代过程。
            black_box(set.into_chunks(block_size).collect::<Vec<_>>());
        })
    });
    group.finish();
}

/// 场景 2: 测试处理一个巨大连续区间的性能。
///
/// 这种情况下，迭代器会在每次 next() 调用中执行“拆分”逻辑，
/// 即弹出一个区间，然后将剩余部分重新插入。
fn bench_chunks_one_large_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("One Large Range");

    // 创建一个包含单个巨大区间的集合
    let mut initial_set = OffsetRangeSet::new();
    // 100 万个元素
    initial_set.insert_range(&OffsetRange::new(0, 1_000_000 - 1));

    // 设置一个较小的块大小，以强制进行大量的拆分操作
    let block_size = 128;

    group.bench_function("split_large_range", |b| {
        b.iter(|| {
            let mut set = initial_set.clone();
            black_box(set.into_chunks(block_size).collect::<Vec<_>>());
        })
    });
    group.finish();
}

/// 场景 3: 测试混合大小区间的性能。
///
/// 这是对两种逻辑（收集和拆分）的综合测试，更贴近实际应用。
fn bench_chunks_mixed_ranges(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Ranges");

    let mut initial_set = OffsetRangeSet::new();
    let mut current_pos = 0;
    for i in 0..500 {
        if i % 5 == 0 {
            // 每 5 个区间，插入一个较大的区间
            let large_range = OffsetRange::new(current_pos, current_pos + 1000);
            initial_set.insert_range(&large_range);
            current_pos += 1001;
        } else {
            // 否则插入一个小区间
            let small_range = OffsetRange::new(current_pos, current_pos + 10);
            initial_set.insert_range(&small_range);
            current_pos += 11;
        }
    }

    let block_size = 1024; // 块大小大于小区间，但小于大区间

    group.bench_function("gather_and_split_mixed", |b| {
        b.iter(|| {
            let mut set = initial_set.clone();
            black_box(set.into_chunks(block_size).collect::<Vec<_>>());
        })
    });
    group.finish();
}

// 使用 criterion 的宏来注册我们的 benchmark 函数
criterion_group!(
    benches,
    bench_chunks_many_small_ranges,
    bench_chunks_one_large_range,
    bench_chunks_mixed_ranges
);
criterion_main!(benches);
