import logging
import random
import time
from datetime import datetime
from itertools import combinations
from typing import Final, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cribbage_scorer import cribbage_scorer
from gymnasium import spaces
from stable_baselines3 import A2C, PPO

CARDS_IN_HAND: Final[int] = 6
CARDS_TO_DISCARD: Final[int] = 2
ALL_SUITS: Final[list[str]] = ["D", "S", "C", "H"]

SUIT_TO_NUMBER: dict[str, int] = {suit: index for index, suit in enumerate(ALL_SUITS)}


class CribbageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, size=5) -> None:

        self.observation_space = spaces.Dict(
            {
                **{"is_dealer": spaces.Discrete(2)},
                **{
                    f"card_{card_index}": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([13, 4]),
                        dtype=np.int64,
                    )
                    for card_index in range(CARDS_IN_HAND)
                },
            }
        )

        self.reward_range = (-30, 60)

        card_indexes: list[int] = list(range(CARDS_IN_HAND))
        self.potential_moves: list = list(combinations(card_indexes, CARDS_TO_DISCARD))

        self.action_space = spaces.Discrete(len(self.potential_moves))

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        current_deck: list[tuple[int, str]] = get_deck()
        random.shuffle(current_deck)

        self.starter_card = current_deck[0]
        del current_deck[0]

        self.current_hand = current_deck[-CARDS_IN_HAND:]
        self.dealt_hand = self.current_hand.copy()
        del current_deck[-CARDS_IN_HAND:]

        self.opponent_crib = current_deck[-CARDS_TO_DISCARD:]
        del current_deck[-CARDS_TO_DISCARD:]

        is_dealer: int = random.choice([0, 1])
        self.is_dealer = is_dealer

        encoded_hand: dict[str, Optional[np.ndarray]] = encode_hand(self.current_hand)

        observation: dict[str, Union[bool, Optional[np.ndarray]]] = {
            **encoded_hand,
            "is_dealer": np.array([is_dealer]),
        }
        info: dict = {}

        return observation, info

    def render(self):
        return f"Hand: {self.current_hand} Starter: {self.starter_card}"

    def get_greedy_hand(self):
        rewards = []
        for action in range(len(self.potential_moves)):
            original_hand, hand_after_discard, reward = self.discard(action)
            self.current_hand = self.dealt_hand.copy()
            rewards.append(reward)

        best_action = np.argmax(rewards)

        return self.discard(best_action)

    def discard(self, action):
        cards_to_discard: tuple[int, int] = self.potential_moves[action]
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
        return self.dealt_hand, hand_after_discard, reward

    def step(self, action) -> tuple:
        logging.debug(f"{action=}")
        logging.debug(f"{self.starter_card=}")
        logging.debug(f"{self.current_hand=}")

        action = action if isinstance(action, np.int64) else action[0]

        crib_cards: list[tuple[int, str]] = self.opponent_crib

        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        logging.debug(f"{cards_to_discard=}")
        for index_to_delete in cards_to_discard:
            crib_cards.append(self.current_hand[index_to_delete])
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

        hand_reward = reward

        crib_reward, crib_msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            crib_cards,
            crib=True,
        )

        if self.is_dealer:
            reward += crib_reward
        else:
            reward -= crib_reward

        logging.debug(
            f"{self.is_dealer=} {hand_reward=} {msg=} {crib_reward=} "
            f"{crib_msg=} {reward=}"
        )

        encoded_hand: dict[str, Optional[np.ndarray]] = encode_hand(self.current_hand)

        observation: dict[str, Union[bool, Optional[np.ndarray]]] = {
            **encoded_hand,
            "is_dealer": np.array([self.is_dealer]),
        }

        terminated: bool = True
        info: dict = {}

        return observation, reward, terminated, False, info


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


def run(model="random", run_steps: int = 5) -> list[int]:

    assert model in ["random", "greedy"] or isinstance(model, PPO)

    current_environment = CribbageEnv()
    observation, info = current_environment.reset()

    rewards: list[int] = []

    for _ in range(run_steps):
        if model == "random":
            # Random action
            action = current_environment.action_space.sample()
        elif model == "greedy":
            current_environment.reset()
            _, _, reward = current_environment.get_greedy_hand()
        else:
            action, _state = model.predict(observation, deterministic=True)

        if model != "greedy":
            observation, reward, terminated, truncated, info = current_environment.step(
                action
            )
        else:
            terminated = False
            truncated = False
            info = {}

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


def train(total_timesteps=10_000):
    current_environment = CribbageEnv()
    model = PPO(
        "MultiInputPolicy",
        current_environment,
        verbose=1,
        n_steps=2048,
        batch_size=2048,
        learning_rate=1e-4,
        tensorboard_log="cribbage_tensorboard_log",
        policy_kwargs={"net_arch": [{"pi": [128, 128, 128], "vf": [128, 128, 128]}]},
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model.save(
        f"saved_models/random_cards_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    return model


def model_close_look(model):
    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    run(
        model,
    )


if __name__ == "__main__":
    print("Getting reference scores...")

    run_steps: int = 100
    total_timesteps: int = 100

    start = time.time()
    model = train(total_timesteps)
    print("Training time:", time.time() - start)

    # model_close_look(model)

    model_rewards: list[int] = run(model, run_steps=run_steps)
    random_rewards: list[int] = run(model="random", run_steps=run_steps)
    greedy_rewards: list[int] = run(model="greedy", run_steps=run_steps)

    print(f"{np.mean(model_rewards)=}")
    print(f"{np.mean(random_rewards)=}")
    print(f"{np.mean(greedy_rewards)=}")

    data: pd.DataFrame = pd.DataFrame(
        {
            "approach": ["random" for _ in range(len(random_rewards))]
            + ["greedy" for _ in range(len(greedy_rewards))]
            + ["model" for _ in range(len(model_rewards))],
            "score": random_rewards + greedy_rewards + model_rewards,
        }
    )

    axes = sns.boxplot(data=data, x="score", y="approach")
    plt.savefig("reward_plot.png")
    plt.show()
