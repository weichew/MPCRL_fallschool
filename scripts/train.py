import gymnasium as gym
import bluesky_gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from safe_rl.rl.wrapper import SafeEnv
import scienceplots          

def make_env():
    bluesky_gym.register_envs()
    return SafeEnv(gym.make("StaticObstacleEnv-v0"))


class SafetyLogger(BaseCallback):
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self.logger.record_mean("safety/corrections", info["corrections"])
            if done:
                self.logger.record_mean("safety/reached", float(info["reached"]))
                self.logger.record_mean("safety/entered", float(info["entered"]))
        return True


def plot_training(run):
    """Two one-column figures: mean episode return and reached fraction against training decisions."""
    log = pd.read_csv(run + "_log/progress.csv")
    x = log["time/total_timesteps"] / 1000
    reached = log["safety/reached"]
    with plt.style.context(["science", "ieee"]):
        # 1. mean episode return
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(x, log["rollout/ep_rew_mean"])
        ax.set_xlabel(r"Training decisions ($\times 10^3$)")
        ax.set_ylabel("Mean episode return")
        fig.savefig(run + "_return.png", dpi=300)
        plt.close(fig)

        # 2. reached fraction
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(x, reached, color="0.7", linestyle="-")                       # per rollout
        ax.plot(x, reached.rolling(5, min_periods=1).mean(), linestyle="-")   # average of 5 rollouts
        ax.set_xlabel(r"Training decisions ($\times 10^3$)")
        ax.set_ylabel("Reached fraction")
        fig.savefig(run + "_reached.png", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    # n_envs = 8
    # run = "results/ppo_duration_list"                                     # one name for model, log and plot
    # env = VecMonitor(SubprocVecEnv([make_env for i in range(n_envs)]))
    # model = PPO("MlpPolicy", env, n_steps=2048 // n_envs, verbose=1, seed=0, gamma=0.999)
    # model.set_logger(configure(run + "_log", ["stdout", "csv"]))          # terminal as before, plus progress.csv
    # model.learn(total_timesteps=80_000, callback=SafetyLogger())
    # model.save(run)
    plot_training("results/ppo_duration_list")