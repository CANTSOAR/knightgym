# STATUS

## Done
- M1 benchmark: see results/benchmark.md. Steady-state, single thread, i9-12950HX: ~5.5M orders/s resting (p50 0.09µs, p99 0.85µs), ~4.4M/s crossing-heavy (p50 0.12µs, p99 0.75µs), ~12.7M ops/s cancel-heavy. cargo test: 10 pass.

## In progress
- M2 PyO3 + Gymnasium env

## Blocked
- none
