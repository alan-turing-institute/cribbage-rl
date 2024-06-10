from environment import CustomEnv
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import numpy as np


from typing import Optional

def run(model: BaseAlgorithm, run_steps:int = 1000) -> None:
    current_environment: CustomEnv = model.get_env()

    if current_environment is not None:
        observation: np.ndarray = current_environment.reset()

        for _ in range(run_steps):
            action, state =  model.predict(observation, deterministic=True)
            observation, reward, done, info =  current_environment.step(action)
            current_environment.render("human")

def main() -> None:
    environment: Env = CustomEnv()
    check_env(environment)

    model: BaseAlgorithm = A2C("CnnPolicy", environment)
    model.learn(total_timesteps=1000)

    run(model)

if __name__ == "__main__":
    main()