//! Criterion throughput benches. Run: cargo bench --bench matching
mod workload;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use workload::*;

const BATCH: usize = 10_000;

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("matching");
    g.throughput(Throughput::Elements(BATCH as u64));
    for mix in [Mix::Resting, Mix::CancelHeavy, Mix::CrossingHeavy] {
        g.bench_function(mix.name(), |b| {
            b.iter_batched(
                || {
                    let mut rng = ChaCha8Rng::seed_from_u64(7);
                    let (book, mut live) = seeded_book(5_000, &mut rng);
                    let ops = build_ops(mix, BATCH, 99, &mut live);
                    (book, ops)
                },
                |(mut book, ops)| {
                    let mut sink = 0usize;
                    for op in ops {
                        sink += apply(&mut book, op);
                    }
                    (book, sink)
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
