import gymnasium as gym
import bluesky as bs

class SafeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def read_state(self):
        base = self.unwrapped
        return {
            "id": bs.traf.id,
            "lat": bs.traf.lat,
            "lon": bs.traf.lon,
            "hdg": bs.traf.hdg,
            "tas": bs.traf.tas,
            "wpt_lat": base.wpt_lat,
            "wpt_lon": base.wpt_lon,
            "polygons": base.obstacle_vertices,
        }
