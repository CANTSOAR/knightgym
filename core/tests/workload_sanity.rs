//! Guards the benchmark workloads: crossing mix must actually trade, cancels must hit.
#[path = "../benches/workload.rs"]
mod workload;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use workload::*;

#[test]
fn crossing_mix_trades_and_cancel_mix_hits() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let (mut book, mut live) = seeded_book(1000, &mut rng);
    let ops = build_ops(Mix::CrossingHeavy, 5000, 2, &mut live);
    let trades: usize = ops.into_iter().map(|o| apply(&mut book, o)).sum();
    assert!(trades > 3000, "crossing mix produced only {trades} fills");

    let (mut book, mut live) = seeded_book(1000, &mut rng);
    let ops = build_ops(Mix::CancelHeavy, 5000, 3, &mut live);
    let n_cancel = ops.iter().filter(|o| matches!(o, Op::Cancel(_))).count();
    let hits: usize = ops.into_iter().map(|o| match o {
        Op::Cancel(_) => apply(&mut book, o),
        o => { apply(&mut book, o); 0 }
    }).sum();
    assert_eq!(hits, n_cancel, "every cancel must find its order");
}
