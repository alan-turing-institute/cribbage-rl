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
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

CARDS_IN_HAND: Final[int] = 6
CARDS_TO_DISCARD: Final[int] = 2
ALL_SUITS: Final[list[str]] = ["D", "S", "C", "H"]
CRIBBAGE_POINTS: str = "cribbage_points"
AGENT_PLAYER: str = "agent"
OPPONENT_PLAYER: str = "opponent"
AGENT_ROUND_SCORE: str = "agent_round_score"
OPPONENT_ROUND_SCORE: str = "opponent_round_score"

TARGET_SCORE: int = 121
AGENT_VICTORY: str = "agent_victory"
IS_DEALER = "is_dealer"
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

        self.reward_range = (-30, 60)

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
        self.start_round()
        return self.encode_observation(), info

    def start_round(self) -> None:
        current_deck: list[tuple[int, str]] = get_deck()
        random.shuffle(current_deck)

        self.starter_card = current_deck[0]
        del current_deck[0]

        self.current_hand = current_deck[-CARDS_IN_HAND:]
        self.dealt_hand = self.current_hand.copy()
        del current_deck[-CARDS_IN_HAND:]

        self.opponent_hand: list[tuple[int, str]] = current_deck[
            -CARDS_IN_HAND:
        ]
        del current_deck[-CARDS_IN_HAND:]

        if self.is_dealer is None:
            self.is_dealer = random.choice([0, 1])
        else:
            if self.is_dealer == 0:
                self.is_dealer = 1
            else:
                self.is_dealer = 0

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

    def get_greedy_hand(self):
        rewards = []
        for action in range(len(self.potential_moves)):
            original_hand, hand_after_discard, reward = self.discard(action)
            self.current_hand = self.dealt_hand.copy()
            rewards.append(reward)

        best_action = np.argmax(rewards)

        return self.discard(best_action)

    def discard(self, action) -> tuple:
        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        for index_to_delete in cards_to_discard:
            self.current_hand[index_to_delete] = None

        hand_after_discard: list[tuple] = [
            card for card in self.current_hand if card is not None
        ]

        reward, msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            hand_after_discard,
            crib=False,
        )
        return self.dealt_hand, hand_after_discard, reward

    def discard_cards(self, action: np.integer) -> tuple[list, list]:

        discarded_cards: list[tuple[int, str]] = []

        cards_to_discard: tuple[int, int] = self.potential_moves[action]
        logging.debug(f"{cards_to_discard=}")
        for index_to_delete in cards_to_discard:
            discarded_cards.append(self.current_hand[index_to_delete])
            self.current_hand[index_to_delete] = None

        hand_after_discard: list[tuple[int, str]] = [
            card for card in self.current_hand if card is not None
        ]

        return hand_after_discard, discarded_cards

    def check_if_there_is_a_winner(self, info):
        terminated = False
        if not self.is_dealer:
            if self.player_score > TARGET_SCORE:
                info[AGENT_VICTORY] = True
                terminated = True
            elif self.opponent_score > TARGET_SCORE:
                info[AGENT_VICTORY] = False
                terminated = True
        else:
            if self.opponent_score > TARGET_SCORE:
                info[AGENT_VICTORY] = False
                terminated = True
            elif self.player_score > TARGET_SCORE:
                info[AGENT_VICTORY] = True
                terminated = True
        return info, terminated

    def step(self, action: Union[np.integer, np.ndarray]) -> tuple:
        # self.start_round()

        logging.debug(f"{action=}")
        logging.debug(f"{self.starter_card=}")
        logging.debug(f"{self.current_hand=}")

        agent_round_score = 0
        opponent_round_score = 0

        cut_scores, cut_msg = cribbage_scorer.cut_calc_score(
            self.starter_card,
            [AGENT_PLAYER, OPPONENT_PLAYER],
            AGENT_PLAYER if self.is_dealer else OPPONENT_PLAYER,
        )
        logging.debug(f"{cut_msg=}")
        logging.debug(f"{cut_scores=}")
        self.player_score += cut_scores[AGENT_PLAYER]
        self.opponent_score += cut_scores[OPPONENT_PLAYER]
        agent_round_score += cut_scores[AGENT_PLAYER]
        opponent_round_score += cut_scores[OPPONENT_PLAYER]

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
        agent_round_score += points_from_hand

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
        self.opponent_score += opponent_hand_points
        opponent_round_score += opponent_hand_points

        crib_cards: list[tuple[int, str]] = (
            self.opponent_crib + discarded_cards
        )
        points_from_crib, crib_msg = cribbage_scorer.show_calc_score(
            self.starter_card,
            crib_cards,
            crib=True,
        )

        info: dict = {}
        if not self.is_dealer:
            self.opponent_score += points_from_crib
            opponent_round_score += points_from_crib
        else:
            self.player_score += points_from_crib
            agent_round_score += points_from_crib

        info, terminated = self.check_if_there_is_a_winner(info)

        reward: int = self.player_score - self.opponent_score
        logging.debug(
            f"{self.is_dealer=} {points_from_hand=} {points_from_crib=}"
            f" {crib_msg=} {reward=}"
        )

        info[CRIBBAGE_POINTS] = self.player_score
        info[IS_DEALER] = self.is_dealer
        info[AGENT_ROUND_SCORE] = agent_round_score
        info[OPPONENT_ROUND_SCORE] = opponent_round_score
        # TODO: Temporal workaround
        # terminated = True
        self.start_round()
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


def run(
    model: Union[str, BaseAlgorithm] = "random", run_steps: int = 5
) -> tuple[list[int], float]:
    print(f"Evaluating {model} for {run_steps} steps")

    assert model in ["random", "greedy"] or isinstance(model, PPO)

    current_environment = CribbageEnv()
    observation, info = current_environment.reset()

    cribbage_points: list[int] = []
    agent_wins = []
    for _ in range(run_steps):
        if model == "random":
            # Random action
            action = current_environment.action_space.sample()
        elif model == "greedy":
            current_environment.reset()
            _, _, points = current_environment.get_greedy_hand()
            cribbage_points.append(points)
        elif isinstance(model, BaseAlgorithm):
            action, _ = model.predict(observation, deterministic=True)

        if model != "greedy":
            observation, reward, terminated, truncated, info = (
                current_environment.step(action)
            )

            if terminated:
                agent_wins.append(info[AGENT_VICTORY])

            cribbage_points.append(info[AGENT_ROUND_SCORE])

        else:
            # CGC: Why are you setting these to False? Resetting triggers
            # shuffling .
            terminated = False
            truncated = False
            info = {}

        # current_environment.render()

        if terminated or truncated:
            observation, info = current_environment.reset()

    win_rate = sum(agent_wins) / len(agent_wins)
    print(f"{model} {sum(agent_wins)=}")
    print(f"{model} {len(agent_wins)=}")

    return cribbage_points, win_rate


def get_deck() -> list[tuple[int, str]]:
    deck: list[tuple[int, str]] = []
    for suit in ALL_SUITS:
        for value in range(1, 14):
            deck.append((value, suit))

    return deck


def train(total_timesteps=10_000):
    print("Starting training...")

    current_environment = CribbageEnv()
    model = PPO(
        "MultiInputPolicy",
        current_environment,
        verbose=1,
        n_steps=2048,
        batch_size=2048,
        learning_rate=1e-4,
        tensorboard_log="cribbage_tensorboard_log",
        policy_kwargs={
            "net_arch": [{"pi": [128, 128, 128], "vf": [128, 128, 128]}]
        },
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model.save(
        "saved_models/"
        f"random_cards_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    evaluation_steps: int = 100_000
    # total_timesteps: int = 100_000
    training_steps: int = 1_000_000

    start = time.time()
    model = train(training_steps)
    print("Training time:", time.time() - start)

    # model_close_look(model)

    model_round_score, model_win_rate = run(model, run_steps=evaluation_steps)
    random_round_score, random_win_rate = run(
        model="random", run_steps=evaluation_steps
    )
    print(f"{model_win_rate=}")
    print(f"{random_win_rate=}")

    # greedy_rewards: list[int] = run(model="greedy", run_steps=run_steps)

    print(f"{np.mean(model_round_score)=}")
    print(f"{np.mean(random_round_score)=}")
    # print(f"{np.mean(greedy_rewards)=}")

    ax = sns.boxplot(
        data=pd.DataFrame(
            {
                "approach": ["random" for _ in range(len(random_round_score))]
                # + ["greedy" for _ in range(len(greedy_rewards))]
                + ["model" for _ in range(len(model_round_score))],
                "score": random_round_score + model_round_score,
            }
        ),
        x="score",
        y="approach",
    )
    plt.savefig("round_scores.png")
    plt.clf()

    axes = sns.barplot(
        data=pd.DataFrame(
            {
                "approach": ["model", "random"],
                "win_rate": [model_win_rate, random_win_rate],
            }
        ),
        x="approach",
        y="win_rate",
    )
    plt.savefig("win_rate.png")
    # plt.show()
