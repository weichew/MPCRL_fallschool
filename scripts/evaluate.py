import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import bluesky_gym
from multiprocessing import Pool
from stable_baselines3 import PPO
import pickle

from safe_rl.rl.wrapper import SafeEnv
from safe_rl.geometry import latlon_to_xy, hdg_to_psi

import scienceplots          

use_pilot = False                 # True: goal-seeking pilot, False: trained PPO
replot_only = True               # True: load saved episodes and only redraw the figures
model_path = "results/ppo_duration_list"
n_episodes = 100
first_seed = 1000
n_workers = 8


def pilot_action(env):
    p = env.p
    s = env.read_state()
    qx_w, qy_w = latlon_to_xy(s["wpt_lat"][0], s["wpt_lon"][0], s["lat"][0], s["lon"][0])
    psi = hdg_to_psi(s["hdg"][0])
    e = np.arctan2(qy_w, qx_w) - psi
    e = (e + np.pi) % (2 * np.pi) - np.pi
    omega_p = np.clip(0.1 * e, -p.omega_max, p.omega_max)
    return np.array([omega_p / p.omega_max])


def run_seeds(seeds):
    bluesky_gym.register_envs()
    env = SafeEnv(gym.make("StaticObstacleEnv-v0"))
    model = None if use_pilot else PPO.load(model_path, device="cpu")
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        sum_progress = sum_barrier = 0.0
        sum_corrections = 0
        s = env.read_state()
        lat0, lon0 = s["lat"][0], s["lon"][0]
        path = [latlon_to_xy(lat0, lon0, lat0, lon0)]
        done = False
        while not done:
            if use_pilot:
                action = pilot_action(env)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            sum_progress += info["r_progress"]
            sum_barrier += info["r_barrier"]
            sum_corrections += info["corrections"]
            done = terminated or truncated
            s = env.read_state()
            path.append(latlon_to_xy(s["lat"][0], s["lon"][0], lat0, lon0))

        if info["reached"]:
            outcome = "reached"
        elif info["entered"]:
            outcome = "entered"
        else:
            outcome = "timeout"
        polygons = [np.array([latlon_to_xy(lat, lon, lat0, lon0) for lat, lon in poly])
                    for poly in s["polygons"]]
        wpt = latlon_to_xy(s["wpt_lat"][0], s["wpt_lon"][0], lat0, lon0)
        episodes.append({"seed": seed, "outcome": outcome, "path": np.array(path),
                         "polygons": polygons, "wpt": wpt,
                         "progress": sum_progress, "barrier": sum_barrier,
                         "corrections": sum_corrections, "decisions": len(path) - 1})
    return episodes


def plot_episodes(eps, prefix, max_plots=6):
    """One figure per episode, same size and same axes box for all."""
    style = ["science", "ieee", {"savefig.bbox": "standard"}]
    with plt.style.context(style):
        for ep in eps[:max_plots]:
            fig = plt.figure(figsize=(1.7, 1.7))
            ax = fig.add_axes([0.26, 0.22, 0.70, 0.74])     # left, bottom, width, height as fractions of the figure
            for poly in ep["polygons"]:
                ax.fill(poly[:, 0], poly[:, 1], color="0.75", linewidth=0)
            path = ep["path"]
            ax.plot(path[:, 0], path[:, 1], color="k", linestyle="-")
            ax.plot(path[0, 0], path[0, 1], "k^", ms=3)                       # start
            ax.plot(ep["wpt"][0], ep["wpt"][1], "*", color="tab:red", ms=5)   # waypoint
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("East (km)")
            ax.set_ylabel("North (km)")
            fig.savefig(f"{prefix}_seed{ep['seed']}.png", dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    name = "pilot" if use_pilot else "ppo"
    episodes_file = f"results/episodes_{name}.pkl"
    if replot_only:
        with open(episodes_file, "rb") as f:
            episodes = pickle.load(f)
    else:
        seeds = list(range(first_seed, first_seed + n_episodes))
        groups = [seeds[i::n_workers] for i in range(n_workers)]
        with Pool(n_workers) as pool:
            results = pool.map(run_seeds, groups)
        episodes = sorted([e for group in results for e in group], key=lambda e: e["seed"])
        with open(episodes_file, "wb") as f:
            pickle.dump(episodes, f)

    counts = {"reached": 0, "entered": 0, "timeout": 0}
    for outcome in ["reached", "timeout"]:
        group = [e for e in episodes if e["outcome"] == outcome]
        if group:
            print(outcome, len(group), "episodes:",
                  "progress", round(np.mean([e["progress"] for e in group]), 2),
                  "barrier", round(np.mean([e["barrier"] for e in group]), 2),
                  "corrections", round(np.mean([e["corrections"] for e in group]), 1),
                  "decisions", round(np.mean([e["decisions"] for e in group]), 1))
    for e in episodes:
        counts[e["outcome"]] += 1
    name = "pilot" if use_pilot else "ppo"
    print(name, counts)
    failed = [e for e in episodes if e["outcome"] != "reached"]
    print("failed seeds:", [e["seed"] for e in failed])

    reached = [ep for ep in episodes if ep["outcome"] == "reached"]
    failed = [ep for ep in episodes if ep["outcome"] != "reached"]
    plot_episodes(reached, f"results/reached_{name}")
    plot_episodes(failed, f"results/failed_{name}")