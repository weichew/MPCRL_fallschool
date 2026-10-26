from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- Time ---
    dt: float = 1.0              # s, simulator step and filter step (Delta t)
    K: int = 10                  # filter steps per policy decision (K)
    max_decisions: int = 200     # decisions per episode before truncation

    # --- Aircraft ---
    omega_max: float = 0.029   # rad/s, largest turn rate (omega_max).  from BlueSky's turn model  
    # speed v is read from bs.traf.tas at reset TODO: change to variable when speed is not constant

    # --- Obstacles ---
    r: float = 0.5       # km, safety margin around each polygon (r).
    kappa: float = 2   # 2/km, smoothing of the union and intersection (kappa). 

    # --- Turning circles ---
    eta: float = 2     # 2/km, smoothing of the union of B_L and B_R (eta). TODO

    # --- Safety filter ---
    beta: float = 0.0073     # 1/s, barrier rate (beta); omega_max / 4, as in the 18 Oct demo
    D_tol: float = 1e-6      # |D| below this counts as zero

    # --- Reward ---
    sigma: float = 0.5        # width of the correction term, as a fraction of omega_max
    d_scale: float = 10.0     # km; progress is divided by this
    c_step: float = 0.01      # cost per decision
    w: float = 0.2            # weight of the mean barrier reward
    r_goal: float = 5.0       # bonus for reaching the waypoint
    d_goal: float = 5.0       # km; reached when closer than this (BlueSky-Gym's value)