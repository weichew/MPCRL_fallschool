import jax.numpy as jnp
from jax.scipy.special import logsumexp


def obstacle_barrier(q, N, b, J, kappa, r):
    """Smooth obstacle barrier h^O(q): positive outside all polygons (with margin r)."""
    # 1. every edge of every polygon: h_mj = n_mj^T q - b_mj - r, shape (M, J_pad)
    h_edges = N @ q - b - r

    # 2. smooth max over each polygon's edges, minus ln(J_m)/kappa, shape (M,)
    h_m = logsumexp(kappa * h_edges, axis=1) / kappa - jnp.log(J) / kappa

    # 3. smooth min over polygons, one number
    return -logsumexp(-kappa * h_m) / kappa