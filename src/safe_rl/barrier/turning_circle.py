import jax.numpy as jnp
from jax.scipy.special import logsumexp
from safe_rl.barrier.obstacle import obstacle_barrier

def circle_values(x, N, b, J, v, omega_max, kappa, r):
    R = v / omega_max
    n = jnp.array([-jnp.sin(x[-1]), jnp.cos(x[-1])])
    c_L = x[:2] + R * n
    c_R = x[:2] - R * n

    B_L = obstacle_barrier(c_L, N, b, J, kappa, r) - R
    B_R = obstacle_barrier(c_R, N, b, J, kappa, r) - R

    return B_L, B_R

def turning_circle_barrier(x, N, b, J, v, omega_max, kappa, r, eta):
    B_L, B_R = circle_values(x, N, b, J, v, omega_max, kappa, r)
    B = logsumexp(eta * jnp.array([B_L, B_R])) / eta - jnp.log(2.0) / eta
    return B