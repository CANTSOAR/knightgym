# Matching-core benchmark

CPU: 12th Gen Intel(R) Core(TM) i9-12950HX, single thread, shared Linux box (load avg ~2 from other services; run under `nice -n 10`). rustc 1.95.0, release profile.

Code under test: `core/src/matching.rs`, an unmodified copy of the Rutgers competition exchange's price-time order book (BTreeMap levels + slab arena + intrusive FIFO lists, i64 fixed-point). Note it calls `SystemTime::now()` once per fill and clones the symbol string per trade; both are left in, so numbers include them.

## Commands (exact)

```
cd core
nice -n 10 cargo bench --bench latency     # per-call latency percentiles + sustained orders/s (custom harness)
nice -n 10 cargo bench --bench matching    # criterion, 10k-op batches on a fresh book
```

## Method
- Book seeded with 5,000 resting orders per side over 200 price levels, 64 owners. 400,000 ops per run, 5 runs (different seeds); median reported. Deterministic (ChaCha8).
- Orders are pre-built so UUID generation is excluded; only `process_limit_order` / `cancel_order` is timed. Each call is timed with `Instant::now()` (adds roughly 20–30 ns of timer overhead to every latency figure).
- Latency is per engine call (cancels count as calls in the cancel-heavy mix).
- Mixes: *resting* = passive limit orders only; *cancel-heavy* = 70% cancels of live orders / 30% new passive; *crossing-heavy* = 70% marketable limit orders priced through the touch (walk up to a few levels, self-trade-prevention cancels possible) / 30% passive. `core/tests/workload_sanity.rs` asserts the crossing mix really trades and every cancel hits.

## Results (latency harness)

| Workload | orders/s (median of 5 runs) | p50 µs | p99 µs | p99.9 µs |
|---|---|---|---|---|
| process_limit_order (resting) | 5.54 M | 0.089 | 0.846 | 1.454 |
| cancel-heavy (70% cancel) | 12.67 M | 0.053 | 0.160 | 0.373 |
| crossing-heavy (70% marketable) | 4.44 M | 0.117 | 0.745 | 3.804 |

## Criterion (fresh book, 10k ops per iteration, cold cache after setup)

    Benchmarking matching/process_limit_order (resting)
    Benchmarking matching/process_limit_order (resting): Warming up for 3.0000 s
    Benchmarking matching/process_limit_order (resting): Collecting 100 samples in estimated 5.7517 s (600 iterations)
    Benchmarking matching/process_limit_order (resting): Analyzing
    matching/process_limit_order (resting)
                            thrpt:  [4.9498 Melem/s 5.0001 Melem/s 5.0450 Melem/s]
    Benchmarking matching/cancel-heavy (70% cancel)
    Benchmarking matching/cancel-heavy (70% cancel): Warming up for 3.0000 s
    Benchmarking matching/cancel-heavy (70% cancel): Collecting 100 samples in estimated 5.6082 s (500 iterations)
    Benchmarking matching/cancel-heavy (70% cancel): Analyzing
    matching/cancel-heavy (70% cancel)
                            thrpt:  [5.2757 Melem/s 5.3028 Melem/s 5.3323 Melem/s]
    Benchmarking matching/crossing-heavy (70% marketable)
    Benchmarking matching/crossing-heavy (70% marketable): Warming up for 3.0000 s
    Benchmarking matching/crossing-heavy (70% marketable): Collecting 100 samples in estimated 6.5399 s (400 iterations)
    Benchmarking matching/crossing-heavy (70% marketable): Analyzing
    matching/crossing-heavy (70% marketable)
                            thrpt:  [3.0448 Melem/s 3.0503 Melem/s 3.0557 Melem/s]

Criterion's cold-start batches are slower than the warm steady-state harness (notably cancel-heavy, 5.3 vs 12.7 M/s) because the book is freshly built per iteration. Quote the warm numbers as steady-state, and say so.

## Caveats
- Single-threaded engine core only. No network, persistence, or Python overhead. The live competition measured ~420 orders/s server-side, which was bound by the bots and API layer, not this core.
- Book depth affects results (BTreeMap lookups); the numbers are for ~10k resting orders.
