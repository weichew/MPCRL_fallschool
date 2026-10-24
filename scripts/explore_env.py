import gymnasium as gym
import bluesky_gym
import random
import numpy as np
import bluesky as bs

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


bluesky_gym.register_envs()
# env = gym.make("StaticObstacleEnv-v0")


# e = env.unwrapped

# # 1. observation
# for k, val in obs.items():
#     print(k, getattr(val, "shape", val))

# # 2. aircraft state
# print(bs.traf.id, bs.traf.lat, bs.traf.lon, bs.traf.hdg, bs.traf.tas)

# # 3. polygons
# print(len(e.obstacle_vertices))          # number of polygons
# for m, poly in enumerate(e.obstacle_vertices):
#     print(m, len(poly), poly[:2])        # number of corners, first two corners
# print(e.obstacle_radius)

# # 4. waypoint
# print(e.wpt_lat, e.wpt_lon)

# # 5. simulator step
# print(bs.sim.simdt)

# import numpy as np
# env.step(np.zeros(2)) # heading change, speed change
# print(bs.sim.simdt)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(7, 7))

# for m, poly in enumerate(e.obstacle_vertices):
#     lats = [c[0] for c in poly]
#     lons = [c[1] for c in poly]
#     ax.fill(lons, lats, alpha=0.3)                 # the polygon
#     # for j, (lat, lon) in enumerate(poly):
#     #     ax.text(lon, lat, str(j), fontsize=9)      # corner numbers

# ax.plot(bs.traf.lon[0], bs.traf.lat[0], "k^", label="aircraft")
# ax.plot(e.wpt_lon[0], e.wpt_lat[0], "r*", markersize=12, label="waypoint")
# ax.set_xlabel("longitude (east)")
# ax.set_ylabel("latitude (north)")
# ax.legend()
# plt.show()

from safe_rl.rl.wrapper import SafeEnv
env = SafeEnv(gym.make("StaticObstacleEnv-v0"))
obs, info = env.reset(seed=SEED)
s0 = env.read_state()
print(s0)
env.step(np.zeros(2))
s1 = env.read_state()
print(s1)
print(s0["lat"], s0["hdg"])