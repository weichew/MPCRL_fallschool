import numpy as np


def make_observation(base, B_L, B_R, B, R):
    """46 numbers: BlueSky-Gym's 43, then B_L/R, B_R/R and B/R."""
    obs = base._get_obs()
    parts = [np.ravel(obs[key]) for key in obs]
    parts.append(np.array([float(B_L) / R, float(B_R) / R, float(B) / R]))
    return np.concatenate(parts).astype(np.float32)