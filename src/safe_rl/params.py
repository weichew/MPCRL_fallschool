from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- Time ---
    dt: float = 1.0              # s, simulator step and filter step (Delta t)
    K: int = 10                  # filter steps per policy decision (K)
    max_decisions: int = 200     # decisions per episode before truncation

    # --- Aircraft ---
    omega_max: float | None = None   # rad/s, largest turn rate (omega_max). TODO: from BlueSky's turn model
    # speed v is read from bs.traf.tas at reset, did bluesky use fuel optimal speed when it is constant?

    # --- Obstacles ---
    r: float | None = None       # km, safety margin around each polygon (r). TODO
    kappa: float | None = None   # 1/km, smoothing of the union and intersection (kappa). TODO

    # --- Turning circles ---
    eta: float | None = None     # 1/km, smoothing of the union of B_L and B_R (eta). TODO

    # --- Safety filter ---
    beta: float | None = None    # 1/s, barrier rate (beta). TODO
    D_tol: float | None = None   # tolerance below which |D| counts as zero. TODO

    # --- Reward ---
    sigma: float | None = None   # width of the barrier-reward term (sigma). TODO
    d_scale: float | None = None # km, distance scale of the progress term (d_scale). TODO
    c_step: float | None = None  # cost per decision (c_step). TODO
    w: float | None = None       # weight of the barrier reward (w). TODO
    r_goal: float | None = None  # reward for reaching the waypoint (r_goal). TODO
    d_goal: float | None = None  # km, distance that counts as reached (d_goal). TODO