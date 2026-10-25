import gymnasium as gym
import bluesky_gym
import random
import numpy as np
import bluesky as bs
from safe_rl.geometry import latlon_to_xy
from bluesky.tools.geo import qdrdist

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
# print(s0)
# env.step(np.zeros(2))
# s1 = env.read_state()
# print(s1)
# print(s0["lat"], s0["hdg"])
lat0 = s0["lat"][0]
lon0 = s0["lon"][0]
# print("aircraft:", latlon_to_xy(lat0, lon0, lat0, lon0))
# wlat = s0["wpt_lat"][0]
# wlon = s0["wpt_lon"][0]
# qx, qy = latlon_to_xy(wlat, wlon, lat0, lon0)
# test = np.sqrt(qx**2 + qy**2)
# bluesky_km = qdrdist(lat0, lon0, wlat, wlon)[1] * 1.852
# print("test:", test, "bluesky:", bluesky_km)
import jax.numpy as jnp
from safe_rl.geometry import build_obstacles
from safe_rl.barrier.obstacle import obstacle_barrier

N, b, J = build_obstacles(s0["polygons"], lat0, lon0)

# # Test A: the aircraft (q = 0) is outside every polygon
# h_air = obstacle_barrier(jnp.zeros(2), N, b, J, kappa=2.0, r=0.5)
# exact = min((-b[m, :J[m]]).max() for m in range(len(J)))   # true min over polygons of max over edges
# print("aircraft:", h_air, "exact:", exact)

# # Test B: the centre of polygon 0 is inside it
# corners = np.array(s0["polygons"][0])
# cx, cy = latlon_to_xy(corners[:, 0].mean(), corners[:, 1].mean(), lat0, lon0)
# h_in = obstacle_barrier(jnp.array([cx, cy]), N, b, J, kappa=1.0, r=0.0)
# print("inside polygon 0:", h_in)

from safe_rl.barrier.turning_circle import circle_values, turning_circle_barrier
from safe_rl.geometry import hdg_to_psi

x0 = jnp.array([0.0, 0.0, hdg_to_psi(s0["hdg"][0])])   # aircraft at origin, its heading
v = s0["tas"][0] / 1000                                 # m/s -> km/s
# print(circle_values(x0, N, b, J, v, 0.029, 2.0, 0.5))
# print(turning_circle_barrier(x0, N, b, J, v, 0.029, 2.0, 0.5, 2.0))

from safe_rl.safety_filter.closed_form import closed_form_filter
from safe_rl.params import Params

p = Params()
print(closed_form_filter(x0, 0.0, N, b, J, v, p))