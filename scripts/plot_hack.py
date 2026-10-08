import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = json.load(open("results/hack_results.json"))
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
order = [("last", "bot"), ("vwap", "bot"), ("mid", "bot"), ("last", "team"), ("vwap", "team"), ("mid", "team")]
get = lambda a, m, s: next(r for r in rows if r["attack"] == a and r["marking"] == m and r["stp"] == s)

fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURF)
ax.set_facecolor(SURF)
x = np.arange(len(order)); w = 0.36
for j, (key, col, lab) in enumerate([("marked_gain", BLUE, "Marked gain (what the reward pays)"),
                                     ("true_gain", ORANGE, "True gain (marked at latent fair value)")]):
    v = [get("wash", m, s)[key][0] for m, s in order]; e = [get("wash", m, s)[key][1] for m, s in order]
    ax.bar(x + (j - .5) * (w + .02), v, w, color=col, label=lab, yerr=e, ecolor=MUTED, capsize=3, error_kw=dict(lw=1))
    for xi, vi in zip(x + (j - .5) * (w + .02), v):
        ax.text(xi, vi + (30 if vi >= 0 else -30), f"{vi:+.0f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=9, color=INK)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"{m} mark\n{s}-level STP" for m, s in order], color=INK)
ax.set_ylabel("Team gain vs. no-attack baseline ($ / episode)", color=MUTED)
ax.set_title("Teammate wash-trading a thin asset: gain collapses with team STP + mid marking", loc="left", color=INK, fontsize=12)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e5e4e0", lw=.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)
ax.annotate("BEFORE\n(competition rules)", (0.3, 1230), ha="center", color=INK, fontsize=9, fontweight="bold")
ax.annotate("AFTER\n(patched)", (5, 150), ha="center", color=INK, fontsize=9, fontweight="bold")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, .8), labelcolor=INK)
ax.set_ylim(-420, 1450)
fig.text(0.01, 0.01, "Mean over 100 paired seeds, 95% CI. 2 teammates x 100 units of a thin asset, 200 steps, wash prints 2.00$ above ref.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig("results/hack_before_after.png", dpi=150, facecolor=SURF)

# markdown table
L = ["| attack | marking | STP | marked gain $ | true gain $ | teammate prints/episode |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['attack']} | {r['marking']} | {r['stp']} | {r['marked_gain'][0]:+.1f} ± {r['marked_gain'][1]:.1f} | {r['true_gain'][0]:+.1f} ± {r['true_gain'][1]:.1f} | {r['wash_fills']:.0f} |")
open("results/hack_results.md", "w").write("\n".join(L) + "\n")
