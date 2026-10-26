import gymnasium as gym
from stable_baselines3 import PPO
from safe_rl.rl.wrapper import SafeEnv
import bluesky_gym
from stable_baselines3.common.callbacks import BaseCallback

class SafetyLogger(BaseCallback):
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self.logger.record_mean("safety/corrections", info["corrections"])  
            if done:
                self.logger.record_mean("safety/reached", float(info["reached"]))
                self.logger.record_mean("safety/entered", float(info["entered"]))
        return True

    
bluesky_gym.register_envs()
env = SafeEnv(gym.make("StaticObstacleEnv-v0"))

model = PPO("MlpPolicy", env, verbose=1, seed=0)
model.learn(total_timesteps=40_000, callback=SafetyLogger())
model.save("results/ppo_plain")