import logging
import random
from itertools import combinations
from typing import Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cribbage_scorer import cribbage_scorer
from gymnasium import spaces
from stable_baselines3 import PPO

CARDS_IN_HAND: int = 6
CARDS_TO_DISCARD: int = 2
ALL_SUITS: list[str] = ["D", "S", "C", "H"]
CRIBBAGE_POINTS: str = "cribbage_points"
AGENT_PLAYER: str = "agent"
OPPONENT_PLAYER: str = "opponent"

TARGET_SCORE: int = 131
AGENT_VICTORY: str = "agent_victory"

SUIT_TO_NUMBER: dict[str, int] = {
    suit: index for index, suit in enumerate(ALL_SUITS)
}


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

        card_indexes: list[int] = list(range(CARDS_IN_HAND))
        self.potential_moves: list = list(
            combinations(card_indexes, CARDS_TO_DISCARD)
        )

        self.action_space = spaces.Discrete(len(self.potential_moves))

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        self.is_dealer: Optional[int] = None
        self.player_score: int = 0
        self.opponent_score: int = 0

        info: dict = {}
        # TODO: Temporal workaround.
        self.start_round()
        return self.encode_observation(), info

    def start_round(self) -> None:
        current_deck: list[tuple[int, str]] = get_deck()
        random.shuffle(current_deck)

        self.starter_card = current_deck[0]
        del current_deck[0]

        self.current_hand = current_deck[-CARDS_IN_HAND:]
        del current_deck[-CARDS_IN_HAND:]

        self.opponent_hand: list[tuple[int, str]] = current_deck[
            -CARDS_IN_HAND:
        ]
        del current_deck[-CARDS_IN_HAND:]

        if self.is_dealer is None:
            self.is_dealer = random.choice([0, 1])
        else:
            self.is_dealer = 0 if self.is_dealer == 1 else 0

    def encode_observation(
        self,
    ) -> dict[str, Union[bool, Optional[np.ndarray]]]:
        encoded_hand: dict[str, Optional[np.ndarray]] = encode_hand(
            self.current_hand
        )

        observation: dict[str, Union[bool, Optional[np.ndarray]]] = {
            **encoded_hand,
            "is_dealer": np.array([self.is_dealer]),
        }

        return observation

    def render(self):
        return f"Hand: {self.current_hand} Starter: {self.starter_card}"

    def discard_cards(self, action: int):

        discarded_cards: list[tuple[int, str]] = []

        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        logging.debug(f"{cards_to_discard=}")
        for index_to_delete in cards_to_discard:
            discarded_cards.append(self.current_hand[index_to_delete])
            self.current_hand[index_to_delete] = None

        hand_after_discard: list = [
            card for card in self.current_hand if card is not None
        ]

        return hand_after_discard, discarded_cards

    def step(self, action) -> tuple:
        # self.start_round()

        logging.debug(f"{action=}")
        logging.debug(f"{self.starter_card=}")
        logging.debug(f"{self.current_hand=}")

        cut_scores, cut_msg = cribbage_scorer.cut_calc_score(
            self.starter_card,
            [AGENT_PLAYER, OPPONENT_PLAYER],
            AGENT_PLAYER if self.is_dealer else OPPONENT_PLAYER,
        )
        logging.debug(f"{cut_msg=}")
        logging.debug(f"{cut_scores=}")
        self.player_score += cut_scores[AGENT_PLAYER]
        self.opponent_score += cut_scores[OPPONENT_PLAYER]

        action = action if isinstance(action, np.integer) else action[0]
        hand_after_discard, discarded_cards = self.discard_cards(action)

        logging.debug(f"{hand_after_discard=}")
        points_from_hand, msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            hand_after_discard,
            crib=False,
        )
        logging.info(f"{msg=}")
        self.player_score += points_from_hand

        self.opponent_crib = random.sample(
            self.opponent_hand, CARDS_TO_DISCARD
        )
        remaining_opponent_hand = set(self.opponent_hand) - set(
            self.opponent_crib
        )
        opponent_hand_points, opponent_message = (
            cribbage_scorer.show_calc_score(
                self.starter_card,
                list(remaining_opponent_hand),
                crib=False,
            )
        )
        logging.debug(
            f"{opponent_hand_points=} {remaining_opponent_hand=} "
            f"{opponent_message=}"
        )
        # TODO: Temporary workaround.
        # self.opponent_score += opponent_hand_points

        crib_cards: list[tuple[int, str]] = (
            self.opponent_crib + discarded_cards
        )
        points_from_crib, crib_msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            crib_cards,
            crib=True,
        )

        terminated: bool = False

        info: dict = {}
        if not self.is_dealer:
            self.opponent_score += points_from_crib
        else:
            self.player_score += points_from_crib

        if self.opponent_score >= TARGET_SCORE:
            info[AGENT_VICTORY] = False
            terminated = True
        elif self.player_score >= TARGET_SCORE:
            info[AGENT_VICTORY] = True
            terminated = True

        reward: int = self.player_score - self.opponent_score
        logging.debug(
            f"{self.is_dealer=} {points_from_hand=} {points_from_crib=}"
            f" {crib_msg=} {reward=}"
        )

        info[CRIBBAGE_POINTS] = self.player_score
        # TODO: Temporal workaround
        terminated = True
        return self.encode_observation(), reward, terminated, False, info


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
            action, _ = model.predict(observation, deterministic=True)

        observation, _, terminated, truncated, info = current_environment.step(
            action
        )
        rewards.append(info[CRIBBAGE_POINTS])
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
        tensorboard_log="cribbage_tensorboard_log",
    )
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
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

    run_steps: int = 1000
    total_timesteps: int = 100_000

    model = train(total_timesteps)

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
