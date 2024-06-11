import gymnasium as gym
import numpy as np
import scoring
from playingcards import Deck


class CardChoice(gym.spaces.MultiDiscrete):

    def __init__(self, nvec, start=None):
        super().__init__(nvec=nvec, start=start)
        self.hand = None
        self.encoded_hand = None

    def sample(self, mask=None):
        if mask is not None:
            raise NotImplementedError()
        else:
            self.hand = self._get_hand()
            self.encoded_hand = self._encode_hand(self.hand)
            return self.encoded_hand

    def _get_hand(self):
        deck = Deck()
        deck.shuffle()
        card_hand = [deck.draw() for _ in range(6)]
        return card_hand

    def _encode_hand(self, hand):
        """
        Return array of shape (6, 2) where each entry is
        (suit, rank) of the card in the hand.
        """
        encoded_hand = []

        for card in hand:
            encoded_hand.append(card.suit["index"])
            encoded_hand.append(card.rank["index"])
        return np.array(encoded_hand).astype(np.int64)


class CribbageEnv(gym.Env):
    def __init__(self):
        # Define Action Space. 15 possible actions: 2 out of 6 cards to place in crib
        self.action_space = gym.spaces.Discrete(15)

        # Define Observation Space is hand of 6
        self.observation_space = CardChoice(
            np.tile(
                [4, 13],
                6,
            ).astype(np.int64),
        )

        self.old_state = [[], [], [], 0]

        # self.observation_space = gym.spaces.MultiDiscrete(np.tile([4, 13], 6,).astype(np.int64))

        self._action_to_discard = {
            0: [0, 1],
            1: [0, 2],
            2: [0, 3],
            3: [0, 4],
            4: [0, 5],
            5: [1, 2],
            6: [1, 3],
            7: [1, 4],
            8: [1, 5],
            9: [2, 3],
            10: [2, 4],
            11: [2, 5],
            12: [3, 4],
            13: [3, 5],
            14: [4, 5],
        }

        self.reward_range = (0, 29)

    def _score_hand(self, cards):
        """Score a hand at the end of a round.

        :param cards: Cards in a single player's hand.
        :return: Points earned by player.
        """
        score = 0
        score_scenarios = [
            scoring.CountCombinationsEqualToN(n=15),
            scoring.HasPairTripleQuad_InHand(),
            scoring.HasStraight_InHand(),
            scoring.HasFlushHand(),
        ]
        for scenario in score_scenarios:
            s, desc = scenario.check(cards[:])
            score += s
            # print("[EOR SCORING] " + desc) if desc else None
        return score

    def step(self, action):
        hand = self.observation_space.hand

        self.old_state[0] = hand

        # print("Hand: ", hand)

        # Define Discard Cards
        discard_cards = self._action_to_discard[action]

        self.old_state[1] = [hand[discard_cards[0]], hand[discard_cards[1]]]

        # Remove the discard cards from the hand
        hand = list(np.delete(hand, discard_cards))
        # print("Hand after discard: ", hand)
        # Calculate the reward
        reward = self._score_hand(hand)
        # reward=np.random.randint(0,29)

        self.old_state[2] = hand
        self.old_state[3] = reward

        # Get Next Hand
        observation = self.observation_space.sample()

        # print(self.observation_space.contains(observation))

        termination = False
        truncation = False
        info = {}
        return observation, reward, termination, truncation, info

    def reset(self, seed: int = 17):
        info: dict = {}
        # print('RESET')
        obs = self.observation_space.sample()
        return obs, info

    def render(self, render_mode="human"):
        pass

    def display_play(self):
        print("Starting Hand: " + str(self.old_state[0]))
        print("Discarded Cards: " + str(self.old_state[1]))
        print("Chosen Cards: " + str(self.old_state[2]))
        print("Reward: " + str(self.old_state[3]))

    def close(self):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    cribbage_env = CribbageEnv()
    cribbage_env.reset()
    print(cribbage_env.step(0))
