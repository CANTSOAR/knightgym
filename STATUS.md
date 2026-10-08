# STATUS

## Done
- M1 benchmark: see results/benchmark.md. Steady-state, single thread, i9-12950HX: ~5.5M orders/s resting (p50 0.09µs, p99 0.85µs), ~4.4M/s crossing-heavy (p50 0.12µs, p99 0.75µs), ~12.7M ops/s cancel-heavy. cargo test: 10 pass.

- M2 PyO3 (maturin) + Gymnasium env + PettingZoo parallel env: DONE. cargo test 14 pass (core 9+1, sim 4), pytest 14 pass. ~84k random-policy steps/s end-to-end from Python. Build: `source .venv/bin/activate && maturin develop --release`.

- M4 reward-hack suite: exploit VERIFIED. Wash trading under last-trade marking + bot-level STP: +$1077/episode marked (±44 95% CI, 100 paired seeds) while true (fair-value) equity falls $93. Patch (team STP + mid marking): marked -$299 ≈ true -$299. Residuals documented: team STP alone still leaves +$388 (house picks off teammate's leftover bid at the print price); sweep-pump +$48 under last/vwap; quote spoofing +$19 under mid (bounded by half-spread). results/hack_results.md, hack_before_after.png, tests/test_hacks.py.

- M3 baselines: DONE. Per 200-step episode (50 held-out seeds): random -26.8 ±4.6, noop 0, scripted MM +1.8 ±0.23, PPO (2M steps, 3 seeds pooled) +2.1 ±0.34. Learning curve: results/learning_curve.png. Caveat: PPO ~ MM, not clearly better; hyperparams tuned once on the eval seeds.
- M4 numbers updated after agent-grouped action ordering: wash/last/bot +1077 marked; wash/last/team +372; wash/mid/team -297 (= true).
- M5 README: DONE (README example executed by tests/test_readme.py).
- Tests: cargo test 14 pass; pytest 19 pass.

## In progress
- nothing; optional extras: learned attacker in PettingZoo env, price-collar patch.

## Blocked
- none
