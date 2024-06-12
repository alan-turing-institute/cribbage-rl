import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import random
from cribbage_scorer import cribbage_scorer
from typing import Optional, Final
from stable_baselines3 import A2C
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

CARDS_IN_HAND: Final[int] = 6
CARDS_TO_DISCARD: Final[int] = 2
ALL_SUITS: Final[list[str]] = ["D", "S", "C", "H"]

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

        card_indexes: list[int] = list(range(CARDS_IN_HAND))
        self.potential_moves: list = list(
            combinations(card_indexes, CARDS_TO_DISCARD)
        )

        self.action_space = spaces.Discrete(len(self.potential_moves))

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        current_deck: list[tuple[int, str]] = get_deck()
        random.shuffle(current_deck)

        self.starter_card = current_deck[0]
        del current_deck[0]

        self.current_hand = current_deck[-CARDS_IN_HAND:]
        del current_deck[-CARDS_IN_HAND:]

        encoded_hand: dict[str, Optional[np.ndarray]] = encode_hand(
            self.current_hand
        )

        info: dict = {}

        return encoded_hand, info

    def render(self):
        return f"Hand: {self.current_hand} Starter: {self.starter_card}"

    def get_greedy_action(self):
        pass

    def discard(self, action):
        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        original_hand = self.current_hand.copy()
        for index_to_delete in cards_to_discard:
                self.current_hand[index_to_delete] = None

        hand_after_discard: list = [
                    card for card in self.current_hand if card is not None
                ]

        reward, msg = cribbage_scorer.show_calc_score(
                    self.starter_card,
                    hand_after_discard,
                    crib=False,
                )
        return original_hand, hand_after_discard, reward

    def step(self, action) -> tuple:
        logging.debug(f"{action=}")
        logging.debug(f"{self.starter_card=}")
        logging.debug(f"{self.current_hand=}")

        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        logging.debug(f"{cards_to_discard=}")
        for index_to_delete in cards_to_discard:
            self.current_hand[index_to_delete] = None

        hand_after_discard: list = [
            card for card in self.current_hand if card is not None
        ]

        logging.debug(f"{hand_after_discard=}")
        reward, msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            hand_after_discard,
            crib=False,
        )

        logging.debug(f"{reward=} {msg=}")

        encoded_hand: dict[str, Optional[np.ndarray]] = encode_hand(
            self.current_hand
        )

        terminated: bool = True
        info: dict = {}

        return encoded_hand, reward, terminated, False, info


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


def get_deck() -> list[tuple[int, str]]:
    deck: list[tuple[int, str]] = []
    for suit in ALL_SUITS:
        for value in range(1, 14):
            deck.append((value, suit))

    return deck


def train(total_timesteps=10_000, model_args={}):
    current_environment = CribbageEnv()
    model = A2C("MultiInputPolicy", current_environment, verbose=1, **model_args)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    return model


def model_close_look(model):
    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    run(
        model,
    )

def get_greedy_action(environment):

    # Score all possible combinations of cards
    potential_scores = []


if __name__ == "__main__":
    print("Getting reference scores...")

    run_steps: int = 1_000
    total_timesteps: int = 10_000

    start = time.time()
    model = train(total_timesteps, model_args={"device":None})
    print("Training time:", time.time() - start)

    # model_close_look(model)

    model_rewards: list[int] = run(model, run_steps=run_steps)
    random_rewards: list[int] = run(run_steps=run_steps)

    print(f"{np.mean(model_rewards)=}")
    print(f"{np.mean(random_rewards)=}")

    data: pd.DataFrame = pd.DataFrame(
        {
            "approach": ["random" for _ in range(len(random_rewards))]
            + ["model" for _ in range(len(model_rewards))],
            "score": random_rewards + model_rewards,
        }
    )

    axes = sns.boxplot(data=data, x="score", y="approach")
    plt.savefig("reward_plot.png")
    plt.show()
