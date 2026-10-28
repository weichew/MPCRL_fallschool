from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- Time ---
    dt: float = 1.0              # s, simulator step and filter step (Delta t)
    K: int = 10                  # filter steps per policy decision (K)
    max_decisions: int = 250     # decisions per episode before truncation

    # --- Aircraft ---
    omega_max: float = 0.029   # rad/s, largest turn rate (omega_max).  from BlueSky's turn model  
    # speed v is read from bs.traf.tas at reset TODO: change to variable when speed is not constant

    # --- Obstacles ---
    r: float = 0.15       # km, safety margin around each polygon (r).
    kappa: float = 6   # 6/km, smoothing of the union and intersection (kappa). 

    # --- Turning circles ---
    eta: float = 5     # 5/km, smoothing of the union of B_L and B_R (eta). 

    # --- Safety filter ---
    beta: float = 0.0073     # 1/s, barrier rate (beta); omega_max / 4
    D_tol: float = 1e-6      # |D| below this counts as zero

    # --- Reward ---
    sigma: float = 0.5        # width of the correction term, as a fraction of omega_max
    d_scale: float = 10.0     # km; progress is divided by this
    c_step: float = 0.01      # cost per decision
    w: float = 0.08            # weight of the mean barrier reward
    r_goal: float = 7.0       # bonus for reaching the waypoint
    d_goal: float = 5.0       # km; reached when closer than this (BlueSky-Gym's value)

    # --- Backup manoeuvres ---
    T_switch: tuple = (30.0, 60.0, 90.0, 120.0, 150.0)   # s, straight-flight durations T_l (0 is B_L, B_R)
    N_s: int = 40                                  # samples per segment; spacing v*120/40 ≈ 0.47 km < r