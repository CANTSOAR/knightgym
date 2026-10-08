import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, collections

BLUE, ORANGE, GREEN, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
res = json.load(open("results/baselines.json"))
runs = collections.defaultdict(list)
for r in csv.DictReader(open("results/ppo_curve.csv")):
    runs[r["run"]].append((int(r["timesteps"]), float(r["mean_pnl"])))
steps = [t for t, _ in next(iter(runs.values()))]
Y = np.array([[y for _, y in v] for v in runs.values()])

fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURF); ax.set_facecolor(SURF)
for y in Y: ax.plot(steps, y, color=BLUE, lw=1, alpha=.35)
ax.plot(steps, Y.mean(0), color=BLUE, lw=2.2)
for name, col, lab, dy in [("scripted_mm", GREEN, "Scripted market maker", -0.55), ("noop", MUTED, "Do nothing", 0.3)]:
    m = res[name]["mean"]; ax.axhline(m, color=col, lw=1.6, ls="--")
    ax.text(steps[-1], m + dy, f" {lab} {m:+.1f}", color=INK, va="center", fontsize=9)
ax.text(steps[-1], Y.mean(0)[-1] + 0.5, f" PPO, mean of {len(Y)} seeds {Y.mean(0)[-1]:+.1f}", color=INK, va="center", fontsize=9, fontweight="bold")
ax.text(0.99, 0.04, f"Random policy: {res['random']['mean']:+.1f} (off chart)", transform=ax.transAxes, ha="right", color=INK, fontsize=9)
ax.set_xlabel("Training timesteps", color=MUTED); ax.set_ylabel("Eval P&L per 200-step episode ($)", color=MUTED)
ax.set_title("PPO vs. baselines on the single-asset market-making env", loc="left", color=INK, fontsize=12)
ax.set_xlim(0, steps[-1] * 1.35); ax.set_ylim(-3, 5.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e5e4e0", lw=.8); ax.tick_params(colors=MUTED)
fig.text(.01, .01, "Thin lines: individual PPO seeds. Eval: 20 fixed held-out seeds per point; baselines on 50 fixed seeds.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig("results/learning_curve.png", dpi=150, facecolor=SURF)
