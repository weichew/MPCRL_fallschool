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
    sigma: float | None = None   # width of the barrier-reward term (sigma). TODO
    d_scale: float | None = None # km, distance scale of the progress term (d_scale). TODO
    c_step: float | None = None  # cost per decision (c_step). TODO
    w: float | None = None       # weight of the barrier reward (w). TODO
    r_goal: float | None = None  # reward for reaching the waypoint (r_goal). TODO
    d_goal: float | None = None  # km, distance that counts as reached (d_goal). TODO