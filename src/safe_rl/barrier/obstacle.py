import jax.numpy as jnp
from jax.scipy.special import logsumexp


def signed_distance(q, N, b, J):
    """Signed distance s_m(q) from q to each convex polygon, shape (M,)."""
    # 1. split each row (nx, ny, px, py) into the normal and the start corner
    n, P = N[..., :2], N[..., 2:]                       # each (M, J_pad, 2)

    # 2. real edges: slot j of polygon m is real if j < J_m
    j = jnp.arange(N.shape[1])                          # (J_pad,)
    valid = j[None, :] < J[:, None]                     # (M, J_pad)

    # 3. end corner of edge j = start corner of edge j+1; the last real edge wraps to 0
    nxt = jnp.where(j[None, :] + 1 >= J[:, None], 0, j[None, :] + 1)          # (M, J_pad)
    P_end = jnp.take_along_axis(P, nxt[..., None].repeat(2, axis=-1), axis=1)  # (M, J_pad, 2)

    # 4. project q onto every edge segment, t in [0, 1] along the edge
    E = P_end - P                                                   # edge vectors p_j+1 - p_j, (M, J_pad, 2)
    t = jnp.sum((q - P) * E, axis=-1) / jnp.maximum(jnp.sum(E * E, axis=-1), 1e-12)
    t = jnp.clip(t, 0.0, 1.0)                                       # (M, J_pad)
    closest = P + t[..., None] * E                                  # closest point on each edge, (M, J_pad, 2)

    # 5. distance to each edge, then to the nearest real edge
    diff = q - closest
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)          # (M, J_pad)
    d_out = jnp.min(jnp.where(valid, dist, 1e6), axis=1)            # (M,)

    # 6. edge values n_mj^T q - b_mj; padded slots never win the max
    h_edges = n @ q - b                                             # (M, J_pad)
    e_max = jnp.max(jnp.where(valid, h_edges, -1e6), axis=1)        # (M,)

    # 7. outside (some edge value > 0): distance; inside: largest edge value (minus the depth)
    return jnp.where(e_max > 0, d_out, e_max)

def obstacle_barrier(q, N, b, J, kappa, r):
    """Smooth obstacle barrier h^O(q): smooth min over polygons of h_m = s_m - r."""
    h_m = signed_distance(q, N, b, J) - r                           # (M,)
    return -logsumexp(-kappa * h_m) / kappa