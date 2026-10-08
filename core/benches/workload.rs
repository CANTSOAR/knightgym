//! Deterministic workloads shared by the criterion and latency benches.
//! Orders are pre-built (UUID generation excluded) so only the engine call is timed.
#![allow(dead_code)]
use knightgym_core::matching::OrderBook;
use knightgym_core::models::{Order, OrderType, Side};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rust_decimal::Decimal;
use uuid::Uuid;

pub enum Op {
    Place(Order),
    Cancel(Uuid),
}

#[derive(Clone, Copy, Debug)]
pub enum Mix {
    /// Passive limit orders only; none cross the spread (engine adds to book).
    Resting,
    /// 70% cancels of live orders / 30% new passive limit orders.
    CancelHeavy,
    /// 70% marketable limit orders that cross the spread / 30% passive replenishment.
    CrossingHeavy,
}

impl Mix {
    pub fn name(self) -> &'static str {
        match self {
            Mix::Resting => "process_limit_order (resting)",
            Mix::CancelHeavy => "cancel-heavy (70% cancel)",
            Mix::CrossingHeavy => "crossing-heavy (70% marketable)",
        }
    }
}

const MID: i64 = 1_000_000; // 100.0000 in price ticks (1e-4)
const OWNERS: i64 = 64;

fn mk(owner: i64, side: Side, px_ticks: i64, qty_lots: i64) -> Order {
    Order::new(
        owner,
        "ABC".into(),
        Decimal::new(px_ticks, 4),
        Decimal::new(qty_lots, 3),
        side,
        OrderType::Limit,
    )
}

/// Seeds a book with `depth` resting orders per side across 200 price levels
/// (bids <= MID-100, asks >= MID+100) and returns the ids so cancels have targets.
pub fn seeded_book(depth: usize, rng: &mut ChaCha8Rng) -> (OrderBook, Vec<Uuid>) {
    let mut book = OrderBook::new();
    let mut live = Vec::with_capacity(depth * 2);
    for _ in 0..depth {
        for side in [Side::Bid, Side::Ask] {
            let off = 100 + rng.gen_range(0..200) * 10;
            let px = if side == Side::Bid { MID - off } else { MID + off };
            let o = mk(rng.gen_range(1..=OWNERS), side, px, rng.gen_range(1..=10) * 1000);
            live.push(o.id);
            book.add_order_to_book(o);
        }
    }
    (book, live)
}

/// Builds `n` ops. Crossing orders are priced through the touch so they always trade
/// if the opposite side is non-empty. Cancels reference ids that were placed earlier
/// (some may already be filled/cancelled, which is realistic: a miss is a cheap no-op,
/// so for cancel-heavy we only target ids not yet consumed by our own bookkeeping).
pub fn build_ops(mix: Mix, n: usize, seed: u64, live_seed: &mut Vec<Uuid>) -> Vec<Op> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let r: f64 = rng.r#gen();
        let owner = rng.gen_range(1..=OWNERS);
        let side = if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
        let qty = rng.gen_range(1..=10) * 1000;
        let passive_px = |rng: &mut ChaCha8Rng| {
            let off = 100 + rng.gen_range(0..200) * 10;
            if side == Side::Bid { MID - off } else { MID + off }
        };
        match mix {
            Mix::Resting => ops.push(Op::Place(mk(owner, side, passive_px(&mut rng), qty))),
            Mix::CancelHeavy => {
                if r < 0.7 && !live_seed.is_empty() {
                    let i = rng.gen_range(0..live_seed.len());
                    ops.push(Op::Cancel(live_seed.swap_remove(i)));
                } else {
                    let o = mk(owner, side, passive_px(&mut rng), qty);
                    live_seed.push(o.id);
                    ops.push(Op::Place(o));
                }
            }
            Mix::CrossingHeavy => {
                if r < 0.7 {
                    // limit through the opposite touch: walks up to ~3 levels
                    let off = rng.gen_range(100..=130);
                    let px = if side == Side::Bid { MID + off } else { MID - off };
                    ops.push(Op::Place(mk(owner, side, px, qty)));
                } else {
                    ops.push(Op::Place(mk(owner, side, passive_px(&mut rng), qty)));
                }
            }
        }
    }
    ops
}

/// Returns (trades, cancels) so the optimizer can't drop the work.
#[inline(always)]
pub fn apply(book: &mut OrderBook, op: Op) -> usize {
    match op {
        Op::Place(o) => {
            let (t, c) = book.process_limit_order(o);
            t.len() + c.len()
        }
        Op::Cancel(id) => book.cancel_order(id).is_some() as usize,
    }
}
