"""Reward-hack audit: wash trading / quote spoofing a thin asset under different marking
rules and self-trade-prevention (STP) scopes.

Team = agents 0 and 1, each holding 100 units of a thin asset. Every cell is paired with a
no-attack baseline on the same seeds (background randomness is identical across the two).
Reported per cell, summed over the team:
  marked gain = reward the env would pay (episode sum of marked-equity changes)
  true gain   = change in equity marked at the latent fair value (what the team really made)
"""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")

from knightgym import _rs, base_config, THIN
from knightgym.attacks import QuoteSpoof, SweepPump, WashTradePair
import json as _json

SEEDS = list(range(500, 600))
MARKINGS = ["last", "vwap", "mid"]
ATTACKS = {"wash": lambda: WashTradePair(markup_cents=200), "sweep": lambda: SweepPump(agent=0), "spoof": lambda: QuoteSpoof(agent=0)}


def cfg(marking, team_stp):
    return base_config(n_agents=2, team_of=[0, 0], team_stp=team_stp, marking=marking, assets=[dict(THIN)],
                       init_inventory=[[100.0], [100.0]], pos_limit=1000.0, episode_len=200)


def episode(c, seed, attack):
    m = _rs.Market(_json.dumps(c))
    m.reset(seed)
    rule = c["marking"]
    start = {r: sum(m.equity(i, r) for i in (0, 1)) for r in (rule, "fair")}
    pol = ATTACKS[attack]() if attack else None
    rewards = 0.0
    n_fill_wash = rejects = stp = 0
    for _ in range(c["episode_len"]):
        out = m.step(pol(m) if pol else [])
        rewards += out["rewards"][0] + out["rewards"][1]
        n_fill_wash += sum(1 for f in out["fills"] if {f[4], f[5]} == {0, 1})
        rejects += len(out["rejects"]); stp += len(out["stp_cancels"])
    true = sum(m.equity(i, "fair") for i in (0, 1)) - start["fair"]
    return dict(marked=rewards, true=true, wash_fills=n_fill_wash, rejects=rejects, stp=stp,
                final_mark=m.marks(rule)[0], final_fair=m.marks("fair")[0])


def ci(x):
    x = np.asarray(x); return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    rows = []
    for attack in ATTACKS:
        for marking in MARKINGS:
            for team_stp in (False, True):
                c = cfg(marking, team_stp)
                base = [episode(c, s, None) for s in SEEDS]
                hit = [episode(c, s, attack) for s in SEEDS]
                dm = [h["marked"] - b["marked"] for h, b in zip(hit, base)]
                dt = [h["true"] - b["true"] for h, b in zip(hit, base)]
                r = dict(attack=attack, marking=marking, stp="team" if team_stp else "bot",
                         marked_gain=ci(dm), true_gain=ci(dt),
                         wash_fills=float(np.mean([h["wash_fills"] for h in hit])),
                         stp_cancels=float(np.mean([h["stp"] for h in hit])),
                         rejects=float(np.mean([h["rejects"] for h in hit])),
                         mark_minus_fair_cents=float(np.mean([(h["final_mark"] - h["final_fair"]) * 100 for h in hit])),
                         n=len(SEEDS))
                rows.append(r)
                print(f"{attack:5s} {marking:5s} stp={r['stp']:4s} marked {r['marked_gain'][0]:9.2f}±{r['marked_gain'][1]:6.2f}"
                      f"  true {r['true_gain'][0]:8.2f}±{r['true_gain'][1]:6.2f}  washfills {r['wash_fills']:6.1f}", flush=True)
    json.dump(rows, open("results/hack_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
