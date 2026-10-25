import numpy as np
Re = 6371 # earth radius in km

def latlon_to_xy(lat, lon, lat0, lon0):
    qy = Re * np.pi/180 * (lat - lat0)
    qx = Re * np.pi/180 * np.cos(np.deg2rad(lat0)) * (lon - lon0) 
    return qx, qy

def hdg_to_psi(hdg):
    psi_deg = 90 - hdg
    return np.deg2rad(psi_deg)

def psi_to_hdg(psi):
    psi_deg = np.rad2deg(psi)
    hdg = 90 - psi_deg
    return hdg % 360

def polygon_edges(qx, qy):
    dx = np.roll(qx, -1) - qx
    dy = np.roll(qy, -1) - qy
    length = np.hypot(dx, dy)

    keep = length > 0.01                 # drop near-duplicate corners
    dx, dy, length = dx[keep], dy[keep], length[keep]
    qx, qy = qx[keep], qy[keep]          # starting corner of each kept edge

    nx = -dy / length                    # outward normal = left of edge
    ny = dx / length
    n = np.stack([nx, ny], axis=1)       # one row (nx, ny) per edge
    b = nx * qx + ny * qy                # b_j = n_j . p_j, one per edge
    return n, b


PAD_SIZES = (8, 16, 32)

def pad_size(j_max):
    """Smallest padding size that fits j_max edges."""
    for size in PAD_SIZES:
        if j_max <= size:
            return size
    return 32 * int(np.ceil(j_max / 32))   # beyond 32: next multiple of 32


def build_obstacles(polygons, lat0, lon0):
    # 1. edges of every polygon, in km around (lat0, lon0)
    edges = []
    for poly in polygons:
        corners = np.array(poly)                       # shape (corners, 2): [lat, lon]
        qx, qy = latlon_to_xy(corners[:, 0], corners[:, 1], lat0, lon0)
        edges.append(polygon_edges(qx, qy))            # (n, b) for this polygon

    # 2. real edge counts and padding size
    J = np.array([len(b) for n, b in edges])
    M, J_pad = len(edges), pad_size(J.max())

    # 3. arrays filled with dummy edges: n = (0, 0), b = 1000
    N = np.zeros((M, J_pad, 2))
    b_all = np.full((M, J_pad), 1000.0)

    # 4. copy each real polygon into its row; the remaining slots stay dummies
    for m, (n, b) in enumerate(edges):
        N[m, :J[m]] = n
        b_all[m, :J[m]] = b
    return N, b_all, J