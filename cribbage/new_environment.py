import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import random
from cribbage_scorer import cribbage_scorer
from typing import Optional
from stable_baselines3 import A2C


CARDS_IN_HAND: int = 6
CARDS_TO_DISCARD: int = 2
ALL_SUITS: list[str] = ["D", "S", "C", "H"]

SUIT_TO_NUMBER: dict[str, int] = {
    suit: index for index, suit in enumerate(ALL_SUITS)
}


class CribbageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, size=5) -> None:

        self.observation_space = spaces.Dict(
            {
                f"card_{card_index}": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([13, 4]),
                    dtype=np.float32,
                )
                for card_index in range(CARDS_IN_HAND)
            }
        )

        self.observation_space["starter"] = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([13, 4]),
            dtype=np.float32,
        )

        card_indexes: list[int] = list(range(CARDS_IN_HAND))
        self.potential_moves: list = list(
            combinations(card_indexes, CARDS_TO_DISCARD)
        )

        self.action_space = spaces.Discrete(len(self.potential_moves))

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        deck: list[tuple[int, str]] = []
        for suit in ALL_SUITS:
            for value in range(1, 14):
                deck.append((value, suit))

        starter_and_hand: list[Optional[tuple[int, str]]] = random.sample(
            deck, CARDS_IN_HAND + 1
        )

        self.starter_card = starter_and_hand[0]
        self.current_hand = starter_and_hand[1:]
        encoded_starter_and_hand: dict[str, Optional[np.ndarray]] = (
            encode_hand(self.current_hand)
        )
        encoded_starter_and_hand["starter"] = encode_card(self.starter_card)

        info: dict = {}

        return encoded_starter_and_hand, info

    def render(self):
        return f"Hand: {self.current_hand} Starter: {self.starter_card}"

    def step(self, action) -> tuple:
        # print(f"{action=}")
        # print(f"{self.starter_card=}")
        # print(f"{self.current_hand=}")

        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        # print(f"{cards_to_discard=}")
        for index_to_delete in cards_to_discard:
            self.current_hand[index_to_delete] = None

        hand_after_discard: list = [
            card for card in self.current_hand if card is not None
        ]

        # print(f"{hand_after_discard=}")
        reward, msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            hand_after_discard,
            crib=False,
        )
        # print(f"{reward=} {msg=}")

        encoded_starter_and_hand: dict[str, Optional[np.ndarray]] = (
            encode_hand(self.current_hand)
        )
        encoded_starter_and_hand["starter"] = encode_card(self.starter_card)

        terminated: bool = True
        info: dict = {}

        return encoded_starter_and_hand, reward, terminated, False, info


def encode_hand(current_hand) -> dict[str, Optional[np.ndarray]]:

    encode_hand: dict[str, Optional[np.ndarray]] = {}
    for index, card in enumerate(current_hand):
        encode_hand[f"card_{index}"] = encode_card(card)

    return encode_hand


def encode_card(card: Optional[tuple[int, str]]) -> Optional[np.ndarray]:

    if card is None:
        return None

    value: int = card[0] - 1
    suit: int = SUIT_TO_NUMBER[card[1]]

    return np.array([value, suit])


def run(model=None, run_steps: int = 5) -> list[int]:
    current_environment = CribbageEnv()
    observation, info = current_environment.reset()

    rewards: list[int] = []

    for _ in range(run_steps):
        if model is None:
            action = current_environment.action_space.sample()
        else:
            action, _state = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = (
            current_environment.step(action)
        )
        rewards.append(reward)
        # current_environment.render()

        if terminated or truncated:
            observation, info = current_environment.reset()

    return rewards


def train(total_timesteps=10_000):
    current_environment = CribbageEnv()
    model = A2C("MultiInputPolicy", current_environment, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    return model


if __name__ == "__main__":
    print("Getting reference scores...")

    run_steps: int = 1000
    total_timesteps: int = 100_000

    model = train(total_timesteps)
    model_rewards: list[int] = run(model, run_steps=run_steps)
    print(f"{np.mean(model_rewards)=}")

    random_rewards: list[int] = run(run_steps=run_steps)
    print(f"{np.mean(random_rewards)=}")
