"""Random policy + scripted market maker + PPO on the same eval seeds.
Usage: python scripts/baselines.py [total_timesteps]   -> results/{baselines.json,ppo_curve.csv,learning_curve.png}"""
import csv, json, sys, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, "scripts")
from common import EVAL_SEEDS, evaluate, evaluate_mm, make_env, random_policy

torch.set_num_threads(2)
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
CURVE_SEEDS = EVAL_SEEDS[:20]
SEEDS = [0, 1, 2]


class Curve(BaseCallback):
    def __init__(self, tag, every=25_000):
        super().__init__()
        self.tag, self.every, self.rows, self.next = tag, every, [], 0

    def _eval(self):
        pnl, _ = evaluate(lambda o, e: self.model.predict(o, deterministic=True)[0], seeds=CURVE_SEEDS)
        self.rows.append((self.tag, self.num_timesteps, pnl.mean(), pnl.std() / np.sqrt(len(pnl))))

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self._eval()
            self.next += self.every
        return True

    def _on_training_end(self):
        self._eval()


def summarize(pnl, inv):
    return dict(mean=float(pnl.mean()), sem=float(pnl.std(ddof=1) / np.sqrt(len(pnl))), std=float(pnl.std(ddof=1)),
                mean_abs_final_inventory=float(np.abs(inv).mean()), n=len(pnl))


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    np.random.seed(0)
    from gymnasium.utils import seeding
    res["random"] = summarize(*evaluate(lambda o, e: e.action_space.sample()))
    res["noop"] = summarize(*evaluate(lambda o, e: np.zeros(5, dtype=int)))
    res["scripted_mm"] = summarize(*evaluate_mm())
    print(res, flush=True)
    rows, ppo_final = [], []
    for sd in SEEDS:
        venv = make_vec_env(lambda: Monitor(make_env()), n_envs=8, seed=sd)
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)  # v2: v1 (plain rewards) collapsed to do-nothing
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, ent_coef=0.003, learning_rate=3e-4,
                    seed=sd, verbose=0, device="cpu", policy_kwargs=dict(net_arch=[128, 128]))
        cb = Curve(f"ppo_seed{sd}")
        model.learn(TOTAL, callback=cb)
        rows += cb.rows
        pnl, inv = evaluate(lambda o, e: model.predict(o, deterministic=True)[0])
        res[f"ppo_seed{sd}"] = summarize(pnl, inv)
        ppo_final.append(pnl)
        print(sd, res[f"ppo_seed{sd}"], f"{time.time()-t0:.0f}s", flush=True)
        model.save(f"results/ppo_seed{sd}")
    allp = np.concatenate(ppo_final)
    res["ppo_all_seeds_pooled"] = dict(mean=float(allp.mean()), sem=float(allp.std(ddof=1) / np.sqrt(len(allp))), n=len(allp))
    res["config"] = dict(total_timesteps=TOTAL, eval_seeds=f"{EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}", curve_seeds=len(CURVE_SEEDS))
    json.dump(res, open("results/baselines.json", "w"), indent=1)
    with open("results/ppo_curve.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "timesteps", "mean_pnl", "sem"]); w.writerows(rows)
