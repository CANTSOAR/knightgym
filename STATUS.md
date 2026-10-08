# STATUS

## Done
- M1 benchmark: see results/benchmark.md. Steady-state, single thread, i9-12950HX: ~5.5M orders/s resting (p50 0.09µs, p99 0.85µs), ~4.4M/s crossing-heavy (p50 0.12µs, p99 0.75µs), ~12.7M ops/s cancel-heavy. cargo test: 10 pass.

- M2 PyO3 (maturin) + Gymnasium env + PettingZoo parallel env: DONE. cargo test 14 pass (core 9+1, sim 4), pytest 14 pass. ~84k random-policy steps/s end-to-end from Python. Build: `source .venv/bin/activate && maturin develop --release`.

## In progress
- M3 baselines (random, scripted MM, PPO)

## Blocked
- none
