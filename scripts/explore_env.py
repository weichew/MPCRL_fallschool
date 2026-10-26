import random
import numpy as np
import gymnasium as gym
import bluesky_gym
import bluesky as bs
import jax.numpy as jnp
import matplotlib.pyplot as plt

from safe_rl.rl.wrapper import SafeEnv
from safe_rl.params import Params
from safe_rl.geometry import latlon_to_xy, hdg_to_psi, psi_to_hdg, build_obstacles
from safe_rl.safety_filter.closed_form import closed_form_filter

# --- 1. set up one scenario ---
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
bluesky_gym.register_envs()
env = SafeEnv(gym.make("StaticObstacleEnv-v0"))
# env.reset(seed=SEED)
# p = Params()

# bs.sim.step()                          # applies BlueSky-Gym's "DT 1": every later step is 1 s
# s = env.read_state()
# v = s["tas"][0] / 1000                 # km/s, constant during the run
# wlat, wlon = s["wpt_lat"][0], s["wpt_lon"][0]

import time

obs, info = env.reset(seed=0)
print("obs:", obs.shape, obs.dtype)

for i in range(3):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    print(round(reward, 3), info)

t0 = time.time()
for i in range(100):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    if terminated or truncated:
        obs, info = env.reset()
print("seconds per decision:", (time.time() - t0) / 100)

# K_GAIN = 0.1                           # 1/s, stand-in policy: turn rate per radian of heading error

# log = {"lat": [], "lon": [], "omega_p": [], "omega": [], "dpsi_err": []}
# outcome = "time limit"

# # --- 2. fly, one filter step per second ---
# for t in range(2000):
#     s = env.read_state()
#     lat0, lon0 = s["lat"][0], s["lon"][0]          # reference point: the aircraft now
#     psi = hdg_to_psi(s["hdg"][0])
#     x = jnp.array([0.0, 0.0, psi])
#     N, b, J = build_obstacles(s["polygons"], lat0, lon0)

#     # stop conditions
#     qx_w, qy_w = latlon_to_xy(wlat, wlon, lat0, lon0)
#     if np.hypot(qx_w, qy_w) < 5.0:
#         outcome = "reached waypoint"
#         break
#     if min((-b[m, :J[m]]).max() for m in range(len(J))) < 0:
#         outcome = "entered a polygon"
#         break

#     # stand-in policy: turn toward the waypoint
#     err = (np.arctan2(qy_w, qx_w) - psi + np.pi) % (2 * np.pi) - np.pi
#     omega_p = float(np.clip(K_GAIN * err, -p.omega_max, p.omega_max))

#     # safety filter
#     omega, A, D = closed_form_filter(x, omega_p, N, b, J, v, p)

#     # command the heading target and fly one second
#     bs.stack.stack(f"HDG {s['id'][0]} {psi_to_hdg(psi + omega * p.dt)}")
#     bs.sim.step()

#     # check: did the aircraft really turn by omega * dt?
#     psi_new = hdg_to_psi(bs.traf.hdg[0])
#     dpsi = (psi_new - psi + np.pi) % (2 * np.pi) - np.pi
#     log["dpsi_err"].append(dpsi - omega * p.dt)

#     log["lat"].append(lat0)
#     log["lon"].append(lon0)
#     log["omega_p"].append(omega_p)
#     log["omega"].append(omega)

# # --- 3. summary ---
# omega_p, omega = np.array(log["omega_p"]), np.array(log["omega"])
# print(outcome, "after", t, "s")
# print("steps where the filter changed the proposal:", np.sum(np.abs(omega - omega_p) > 1e-9))
# print("largest turn error (rad):", np.max(np.abs(log["dpsi_err"])))

# # --- 4. plot the path ---
# fig, ax = plt.subplots(figsize=(7, 7))
# for poly in s["polygons"]:
#     corners = np.array(poly)
#     ax.fill(corners[:, 1], corners[:, 0], alpha=0.3)
# ax.plot(log["lon"], log["lat"], "k-", label="path")
# ax.plot(log["lon"][0], log["lat"][0], "k^", label="start")
# ax.plot(wlon, wlat, "r*", markersize=12, label="waypoint")
# ax.set_xlabel("longitude (east)")
# ax.set_ylabel("latitude (north)")
# ax.legend()
# plt.show()