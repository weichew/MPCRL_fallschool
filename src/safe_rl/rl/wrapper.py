import random
import numpy as np
import gymnasium as gym
import bluesky as bs
import jax
import jax.numpy as jnp

from safe_rl.params import Params
from safe_rl.geometry import latlon_to_xy, hdg_to_psi, psi_to_hdg, build_obstacles
from safe_rl.barrier.turning_circle import circle_values, turning_circle_barrier
from safe_rl.safety_filter.closed_form import closed_form_filter
from safe_rl.rl.reward import barrier_reward, decision_reward
from safe_rl.rl.observation import make_observation

circle_values_fast = jax.jit(circle_values)
barrier_fast = jax.jit(turning_circle_barrier)

class SafeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.p = Params()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(46, ), dtype=np.float32)
        self.log = None

    def read_state(self):
        base = self.unwrapped
        return {
            "id": bs.traf.id,
            "lat": bs.traf.lat,
            "lon": bs.traf.lon,
            "hdg": bs.traf.hdg,
            "tas": bs.traf.tas,
            "wpt_lat": base.wpt_lat,
            "wpt_lon": base.wpt_lon,
            "polygons": base.obstacle_vertices,
        }

    def _local(self):
        s = self.read_state()
        lat0, lon0 = s["lat"][0], s["lon"][0]
        psi = hdg_to_psi(s["hdg"][0])
        x = jnp.array([0.0, 0.0, psi])
        N, b, J = build_obstacles(s["polygons"], lat0, lon0)
        qx_w, qy_w = latlon_to_xy(s["wpt_lat"][0], s["wpt_lon"][0], lat0, lon0)
        d = float(np.hypot(qx_w, qy_w)) # distance from the aircraft (the origin) to the waypoint
        return s, x, N, b, J, d

    def _observe(self, x, N, b, J):
        p = self.p
        s = self.read_state()
        B_L, B_R = circle_values_fast(x, N, b, J, self.v, p.omega_max, p.kappa, p.r)
        B = barrier_fast(x, N, b, J, self.v, p.omega_max, p.kappa, p.r, p.eta)
        obs = make_observation(self.unwrapped, B_L, B_R, B, self.R)
        return obs, float(B_L), float(B_R)

    def reset(self, seed=None, options=None):
        p = self.p
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        while True:
            self.env.reset(seed=seed, options=options)
            if bs.sim.simdt != p.dt:
                bs.stack.stack(f"HDG {bs.traf.id[0]} {bs.traf.hdg[0]:.6f}")
                bs.sim.step()
            self.v = float(bs.traf.tas[0]) / 1000
            self.R = self.v / p.omega_max
            s, x, N, b, J, d = self._local()
            obs, B_L, B_R = self._observe(x, N, b, J)
            B = float(turning_circle_barrier(x, N, b, J, self.v, p.omega_max, p.kappa, p.r, p.eta))
            if B >= 0:
                break
            seed = None
        self.k = 0
        self.d_prev = d
        return obs, {}

    def step(self, action):
        p = self.p
        omega_p = p.omega_max * action[0]
        rB = []
        corrections = 0
        reached = entered = False
        for i in range(p.K):
            s, x, N, b, J, d = self._local()
            omega, A, D = closed_form_filter(x, omega_p, N, b, J, self.v, p)
            if self.log is not None:
                B_L, B_R = circle_values_fast(x, N, b, J, self.v, p.omega_max, p.kappa, p.r)
                self.log.append({"lat": s["lat"][0], "lon": s["lon"][0],
                                "B_L": float(B_L), "B_R": float(B_R),
                                "omega_p": omega_p, "omega": omega})
            rB.append(barrier_reward(A, D, omega_p, omega, self.v, p)) 
            corrections += abs(omega - omega_p) > 1e-9 
            psi = float(x[2])
            hdg_cmd = psi_to_hdg(psi + omega * p.dt)
            bs.stack.stack(f"HDG {s['id'][0]} {hdg_cmd:.6f}")
            bs.sim.step()
            entered = bool(self.unwrapped._check_intrusion()[1])
            d = self._local()[5]
            reached = d < p.d_goal
            if entered or reached:
                break
        s, x, N, b, J, d = self._local()
        obs, B_L, B_R = self._observe(x, N, b, J)
        reward = decision_reward(self.d_prev, d, float(np.mean(rB)), reached, p)
        rB_mean = float(np.mean(rB))
        reward = decision_reward(self.d_prev, d, rB_mean, reached, p)
        r_progress = (self.d_prev - d) / p.d_scale
        r_barrier = p.w * rB_mean
        self.d_prev = d
        self.k += 1
        terminated = reached or entered
        truncated = self.k >= p.max_decisions                              # k has reached max_decisions
        info = {"reached": reached, "entered": entered, "corrections": int(corrections),
                "distance": d, "r_progress": r_progress, "r_barrier": r_barrier}
        return obs, float(reward), terminated, truncated, info
