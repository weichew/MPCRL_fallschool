import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from safe_rl.barrier.obstacle import obstacle_barrier

from safe_rl.params import Params

p_ = Params()
T_SWITCH = jnp.array(p_.T_switch)            # (N_T,) s
FRAC = jnp.linspace(0.0, 1.0, p_.N_s + 1)    # (N_s+1,) position along each segment

def backup_points(x, v):
    q, psi = x[:2], x[2]
    e = jnp.array([jnp.cos(psi), jnp.sin(psi)])
    dist = v * T_SWITCH[:, None] * FRAC[None, :]  # (N_T, N_s+1) km along each segment
    s = q + dist[..., None] * e                   # (N_T, N_s+1, 2) sample points
    q_T = q + v * T_SWITCH[:, None] * e           # (N_T, 2) end points
    return s, q_T

def circle_values(x, N, b, J, v, omega_max, kappa, r):
    R = v / omega_max
    n = jnp.array([-jnp.sin(x[-1]), jnp.cos(x[-1])])
    c_L = x[:2] + R * n
    c_R = x[:2] - R * n

    B_L = obstacle_barrier(c_L, N, b, J, kappa, r) - R
    B_R = obstacle_barrier(c_R, N, b, J, kappa, r) - R

    return B_L, B_R

def straight_turn_values(x, N, b, J, v, omega_max, kappa, r):
    s, q_T = backup_points(x, v)                  # (N_T, N_s+1, 2), (N_T, 2)
    delta = v * T_SWITCH / p_.N_s                 # (N_T,) spacing of the samples

    h_one = lambda q: obstacle_barrier(q, N, b, J, kappa, r)
    h_s = jax.vmap(jax.vmap(h_one))(s)            # (N_T, N_s+1) h^O at every sample
    seg = h_s - delta[:, None] / 2                # sample terms h^O - delta/2

    psi = jnp.full((T_SWITCH.shape[0], 1), x[-1])
    x_T = jnp.concatenate([q_T, psi], axis=1)     # (N_T, 3) end states, heading unchanged
    circ = lambda xe: circle_values(xe, N, b, J, v, omega_max, kappa, r)
    B_L_end, B_R_end = jax.vmap(circ)(x_T)        # (N_T,) each

    terms_L = jnp.concatenate([seg, B_L_end[:, None]], axis=1)
    terms_R = jnp.concatenate([seg, B_R_end[:, None]], axis=1)
    B_Ll = -logsumexp(-kappa * terms_L, axis=1) / kappa
    B_Rl = -logsumexp(-kappa * terms_R, axis=1) / kappa
    return B_Ll, B_Rl

def all_pieces(x, N, b, J, v, omega_max, kappa, r):
    B_L, B_R = circle_values(x, N, b, J, v, omega_max, kappa, r)
    B_Ll, B_Rl = straight_turn_values(x, N, b, J, v, omega_max, kappa, r)
    return jnp.concatenate([jnp.array([B_L, B_R]), B_Ll, B_Rl])   # (2 N_T + 2,)

def turning_circle_barrier(x, N, b, J, v, omega_max, kappa, r, eta):
    pieces = all_pieces(x, N, b, J, v, omega_max, kappa, r)
    return logsumexp(eta * pieces) / eta - jnp.log(pieces.shape[0]) / eta