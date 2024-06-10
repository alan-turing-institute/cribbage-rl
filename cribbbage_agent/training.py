from environment import CustomEnv
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional

def run(model: BaseAlgorithm) -> None:
    current_environment: Optional[VecEnv] = model.get_env()

def main() -> None:
    environment: Env = CustomEnv()
    check_env(environment)

    model: BaseAlgorithm = A2C("CnnPolicy", environment)
    model.learn(total_timesteps=1000)

    run(model)

if __name__ == "__main__":
    main()