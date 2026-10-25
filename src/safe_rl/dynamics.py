import jax.numpy as jnp


def f(x, v):
    """Drift: how the state changes with no turning. x = (qx, qy, psi), v in km/s."""
    psi = x[2]
    return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.0])


def g(x):
    """Input direction: how the turn rate omega enters the state change."""
    return jnp.array([0.0, 0.0, 1.0])