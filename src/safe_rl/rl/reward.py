import numpy as np


def barrier_reward(A, D, omega_p, omega, v, p):
    """Barrier reward r^B_t for one second.

    A, D     : from closed_form_filter (km/s and km/rad)
    omega_p  : proposed turn rate (rad/s)
    omega    : turn rate after the filter (rad/s)
    v        : speed (km/s)
    """
    slack = A + D * omega_p                          # s_t: how well the proposal meets the condition
    violation = min(slack / v, 0.0)                  # 0 if met, negative if violated
    correction = np.exp(-(omega_p - omega) ** 2 / (p.sigma ** 2 * p.omega_max ** 2)) - 1.0
    return violation + correction
    # return violation


def decision_reward(d_prev, d_now, rB_mean, reached, p):
    """Reward R_{k+1} for one decision.

    d_prev, d_now : distance to the waypoint before and after the decision (km)
    rB_mean       : mean of barrier_reward over the decision's seconds
    reached       : True if the aircraft came within d_goal of the waypoint
    """
    progress = (d_prev - d_now) / p.d_scale
    return progress - p.c_step + p.w * rB_mean + p.r_goal * float(reached)