import numpy as np
Re = 6371 # earth radius in km

def latlon_to_xy(lat, lon, lat0, lon0):
    qy = Re * np.pi/180 * (lat - lat0)
    qx = Re * np.pi/180 * np.cos(np.deg2rad(lat0)) * (lon - lon0) 
    return qx, qy