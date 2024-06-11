import gymnasium as gym
import numpy as np
from environment import CribbageEnv
from gymnasium import Env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


def run(model: BaseAlgorithm, run_steps: int = 5, render_mode="human") -> None:
    current_environment: Env = model.get_env()

    if current_environment is not None:
        observation: np.ndarray = current_environment.reset()

        for _ in range(run_steps):
            action, state = model.predict(observation, deterministic=True)
            print(f"-------------- {_} --------------")
            current_environment.step(action)
            current_environment.env_method("display_play")

        reward_vals = []

        for _ in range(1000):
            action, state = model.predict(observation, deterministic=True)
            observation, reward, done, info = current_environment.step(action)
            reward_vals.append(reward)

        print(f"Average Reward: {np.mean(reward_vals)}")

    return model


def train(current_environment: Env, total_timesteps=1000, verbose=1) -> BaseAlgorithm:
    current_environment.reset()

    model: BaseAlgorithm = PPO("MlpPolicy", current_environment, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    return model


def main() -> None:
    environment: Env = CribbageEnv()

    # environment: Env = gym.make("LunarLander-v2", render_mode="human")
    check_env(environment)

    total_timesteps: int = 1_000_000
    model: BaseAlgorithm = train(environment, total_timesteps)

    run(model)


if __name__ == "__main__":
    main()
