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
bluesky_gym.register_envs()
env = SafeEnv(gym.make("StaticObstacleEnv-v0"))

from operator import attrgetter


KT = 0.514444                                    # m/s per knot

obs, info = env.reset(seed=SEED)
acid = bs.traf.id[0]
t0, lat0 = bs.sim.simt, bs.traf.lat[0]
target_kt = bs.traf.cas[0] / KT - 30
bs.stack.stack(f"SPD {acid} {target_kt:.1f}")
bs.sim.step()
bs.sim.step()
print("sim time advanced:", bs.sim.simt - t0, "s, lat moved:", bs.traf.lat[0] - lat0)

for label, path in [
    ("selected speed (cas, m/s)", "selspd"),
    ("autopilot target tas (m/s)", "ap.tas"),
    ("target tas after limits (m/s)", "aporasas.tas"),
    ("perf vmin (cas, m/s)", "perf.vmin"),
    ("perf vmax (cas, m/s)", "perf.vmax"),
    ("flight phase", "perf.phase"),
    ("acceleration (m/s^2)", "ax"),
]:
    try:
        print(label, attrgetter(path)(bs.traf)[0])
    except AttributeError:
        print(label, "not found")