import jax
from safe_rl.barrier.turning_circle import turning_circle_barrier, circle_values
from safe_rl.dynamics import f, g
barrier_and_grad = jax.jit(jax.value_and_grad(turning_circle_barrier))

def closed_form_filter(x, omega_p, N, b, J, v, p):
    B, gradB = barrier_and_grad(x, N, b, J, v, p.omega_max, p.kappa, p.r, p.eta)
    LfB = gradB @ f(x, v)
    LgB = gradB @ g(x)
    A = LfB + p.beta * B
    D = LgB
    A = float(LfB + p.beta * B)
    D = float(LgB)
    if D > p.D_tol:                        # case 1
        lo = max(-p.omega_max, -A / D)
        hi = p.omega_max
    elif D < -p.D_tol:                     # case 2
        lo = -p.omega_max
        hi = min(p.omega_max, -A / D)
    elif A >= 0:                           # case 3, everything allowed
        lo, hi = -p.omega_max, p.omega_max
    else:                                  # case 3, nothing allowed
        lo, hi = 1.0, -1.0                 # lo > hi marks an empty interval
    if lo <= hi: 
        omega = min(max(omega_p, lo), hi)
    else:
        B_L, B_R = circle_values(x, N, b, J, v, p.omega_max, p.kappa, p.r)
        if B_L >= B_R:
            omega = p.omega_max
        else:
            omega = -p.omega_max
    return omega, A, D