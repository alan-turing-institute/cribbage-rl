import numpy as np
import gymnasium as gym
from playingcards import Deck 
import scoring

class CardChoice(gym.spaces.MultiDiscrete):

    def __init__(self, nvec):
        super().__init__(nvec)
        self.hand = None
        self.encoded_hand=None

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
            encoded_hand.append([card.suit['index'], card.rank['index']])
        return np.array(encoded_hand)


class CribbageEnv(gym.Env):
    def __init__(self):
        # Define Action Space. 15 possible actions: 2 out of 6 cards to place in crib
        self.action_space = gym.spaces.Discrete(15)

        # Define Observation Space is hand of 6
        self.observation_space = CardChoice(np.tile([[4], [13]], 6,).T)

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
            14: [4, 5]
        }

        self.reward_range = (0, 29)


    def _score_hand(self, cards):
        """Score a hand at the end of a round.

        :param cards: Cards in a single player's hand.
        :return: Points earned by player.
        """
        score = 0
        score_scenarios = [scoring.CountCombinationsEqualToN(n=15),
                           scoring.HasPairTripleQuad(), scoring.HasStraight_InHand(), scoring.HasFlush()]
        for scenario in score_scenarios:
            s, desc = scenario.check(cards[:])
            score += s
            print("[EOR SCORING] " + desc) if desc else None
        return score

    def step(self, action):

        hand = self.observation_space.hand

        print("Hand: ", hand)

        # Define Discard Cards
        discard_cards = self._action_to_discard[action]

        # Remove the discard cards from the hand
        hand = list(np.delete(hand, discard_cards))
        print("Hand after discard: ", hand)
        # Calculate the reward
        reward = self._score_hand(hand)

        # Get Next Hand 
        observation = self.observation_space.sample()

        termination = False
        truncation = False
        info = {}
        return observation, reward, termination, truncation, info

    def reset(self):
        return self.observation_space.sample()
        
    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

if __name__ == "__main__":
    cribbage_env = CribbageEnv()
    cribbage_env.reset()
    print(cribbage_env.step(0))