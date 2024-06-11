from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import numpy as np
from environment import CribbageEnv


def run(
    model: BaseAlgorithm, run_steps: int = 1000, render_mode="human"
) -> None:
    current_environment: Env = model.get_env()

    if current_environment is not None:
        observation: np.ndarray = current_environment.reset()

        for _ in range(run_steps):
            action, state = model.predict(observation, deterministic=True)
            observation, reward, termination, truncation, info = (
                current_environment.step(action)
            )
            current_environment.render(render_mode=render_mode)


def train(
    current_environment: Env, total_timesteps=1000, verbose=1
) -> BaseAlgorithm:
    current_environment.reset()

    model: BaseAlgorithm = PPO(
        "MlpPolicy", current_environment, verbose=verbose
    )
    model.learn(total_timesteps=total_timesteps)

    return model


def main() -> None:
    environment: Env = CribbageEnv()

    # environment: Env = gym.make("LunarLander-v2", render_mode="human")
    check_env(environment)

    # total_timesteps: int = 100000
    total_timesteps: int = 10000
    model: BaseAlgorithm = train(environment, total_timesteps)

    run(model)


if __name__ == "__main__":
    main()
