//! Per-call latency (p50/p99/p99.9) and sustained orders/s for each workload.
//! Run: cargo bench --bench latency
mod workload;
use knightgym_core::matching::OrderBook;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use workload::*;

const OPS: usize = 400_000;
const DEPTH: usize = 5_000;
const RUNS: usize = 5;

fn pct(sorted: &[u64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p) as usize] as f64 / 1000.0
}

fn main() {
    println!("workload,run,ops,ops_per_s,p50_us,p99_us,p999_us");
    for mix in [Mix::Resting, Mix::CancelHeavy, Mix::CrossingHeavy] {
        for run in 0..RUNS {
            let mut rng = ChaCha8Rng::seed_from_u64(run as u64);
            let (mut book, mut live): (OrderBook, _) = seeded_book(DEPTH, &mut rng);
            let ops = build_ops(mix, OPS, 1000 + run as u64, &mut live);
            let mut lat: Vec<u64> = Vec::with_capacity(OPS);
            let mut sink = 0usize;
            let t0 = Instant::now();
            for op in ops {
                let s = Instant::now();
                sink += apply(&mut book, op);
                lat.push(s.elapsed().as_nanos() as u64);
            }
            let total = t0.elapsed().as_secs_f64();
            std::hint::black_box(sink);
            lat.sort_unstable();
            println!(
                "{},{},{},{:.0},{:.3},{:.3},{:.3}",
                mix.name(), run, OPS, OPS as f64 / total,
                pct(&lat, 0.5), pct(&lat, 0.99), pct(&lat, 0.999)
            );
        }
    }
}
