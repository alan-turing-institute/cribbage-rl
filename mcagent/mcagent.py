import random
import heapq
import math
import os
import json

from copy import deepcopy
from itertools import combinations
import numpy as np

SPADES = 0
HEARTS = 1
CLUBS = 2
DIAMONDS = 3


CARD_VALUE_STRINGS = {
    11:"J",
    12:"Q",
    13:"K",
    1:"A",
    2:"2",
    3:"3",
    4:"4",
    5:"5",
    6:"6",
    7:"7",
    8:"8",
    9:"9",
    10:"10"
}

SUIT_STRINGS = {
    SPADES:"♠",
    HEARTS:"♥",
    CLUBS:"♣",
    DIAMONDS:"♦"
}


def card_to_string(card):
    if card[0] in CARD_VALUE_STRINGS and card[1] in SUIT_STRINGS:
        # return CARD_VALUE_STRINGS[card[0]] +" of "+SUIT_STRINGS[card[1]]
        return CARD_VALUE_STRINGS[card[0]]+SUIT_STRINGS[card[1]]
    else:
        raise Exception("Invalid card: "+card)

def peg_val(card):
    return 10 if card[0]>10 else card[0]

def score_hand(hand4cards, cutcard, is_crib=False):
    """
    Returns the total point value of a 4 card hand with the given cut card
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :param is_crib: if the hand being scored is the crib
    :return: integer point value of the hand
    """
    total_points = 0
    total_points += right_jack(hand4cards,cutcard)
    total_points += flush(hand4cards, cutcard, is_crib)

    sorted5cards=sort_cards(hand4cards,cutcard)

    total_points += two_card_fifteens(sorted5cards)
    total_points += three_card_fifteens(sorted5cards)
    total_points += four_card_fifteens(sorted5cards)
    total_points += five_card_fifteens(sorted5cards)
    total_points += runs(sorted5cards)
    total_points += pairs(sorted5cards)

    return total_points


def sort_cards(hand4cards,cutcard):
    """
    puts the hand of 4 cards and the cut card into one sorted hand
    :param hand4cards: 4 cards in the player's hand
    :param cutcard: cut card
    :return: sorted five card hand
    """
    hand_queue = []

    for c in hand4cards:
        heapq.heappush(hand_queue, c)
    heapq.heappush(hand_queue,cutcard)
    sorted5cards = heapq.nsmallest(5, hand_queue)
    return sorted5cards


def right_jack(hand4cards, cutcard):
    """
    Returns the point value from right jacks in the given hand
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :return: 1 point if the hand contains the right jack, 0 otherwise
    """
    points = 0
    # right jack
    for card in hand4cards:
        if card[0] == 11 and cutcard[1] == card[1]:  # if card in hand is a Jack and its suit matches the cut card
            points += 1
    return points


def flush(hand4cards, cutcard, is_crib):
    """
    Returns the point value from flushes in the given hand
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :return: points from flushes
    """
    points=0
    # flushes
    if hand4cards[0][1] == hand4cards[1][1] == hand4cards[2][1] == hand4cards[3][1]:
        points += 4
        if hand4cards[0][1] == cutcard[1]:
            points += 1
    if is_crib:
        if points==4:
            points=0

    return points



def two_card_fifteens(sorted5cards):
    """
    Returns the point value of pairs of cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from two card 15's
    """
    points=0
    index_combinations2 = combinations([0,1,2,3,4], 2)
    for combination in list(index_combinations2):
        card1 = sorted5cards[combination[0]]
        value1=peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2=peg_val(card2)
        if value1 + value2 == 15:
            points += 2
    return points

def three_card_fifteens(sorted5cards):
    """
    Returns the point value of 3 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from three card 15's
    """
    points=0
    index_combinations3 = combinations([0, 1, 2, 3, 4], 3)
    for combination in list(index_combinations3):
        card1 = sorted5cards[combination[0]]
        value1 = peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = peg_val(card3)
        if value1 + value2 + value3 == 15:
            points += 2
    return points

def four_card_fifteens(sorted5cards):
    """
    Returns the point value of 4 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from four card 15's
    """
    points=0
    index_combinations4 = combinations([0, 1, 2, 3, 4], 4)
    for combination in list(index_combinations4):
        card1 = sorted5cards[combination[0]]
        value1 = peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = peg_val(card3)
        card4 = sorted5cards[combination[3]]
        value4=peg_val(card4)
        if value1 + value2 + value3 + value4 == 15:
            points += 2
    return points

def five_card_fifteens(sorted5cards):
    """
    Returns the point value of 5 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from five card 15's
    """
    points=0
    sum=0
    for i in range(5):
        card=sorted5cards[i]
        sum+=peg_val(card)
    if sum==15:
        points+=2
    return points


def runs(sorted5cards):
    """
    Returns the point value from runs
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from runs
    """
    points=0
    for start_index in range(3):
        next_index=start_index+1
        consecutive_cards_count = 1
        duplicates_count = 0
        while next_index<5:
            if sorted5cards[start_index][0] == sorted5cards[next_index][0]:
                duplicates_count += 1
            elif sorted5cards[start_index][0] == sorted5cards[next_index][0] - 1:
                consecutive_cards_count += 1
            else:
                break
            start_index = next_index
            next_index += 1
        multiplier = 1
        if duplicates_count > 0:
            multiplier = duplicates_count*2
        if consecutive_cards_count >= 3:
            points += multiplier * consecutive_cards_count
            break
    return points

def pairs(sorted5cards):
    """
    Returns the point value from pairs (includes 3 of a kind and 4 of a kind)
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from pairs
    """
    points=0
    start_card_index = 0
    while start_card_index < 4:
        index = start_card_index + 1
        for i in range(index, 5):
            if sorted5cards[start_card_index][0] == sorted5cards[i][0]:
                points += 2
        start_card_index += 1
    return points

class MCAgent(object):
    
    def __init__(self):
        cwd_name = os.path.dirname(os.path.realpath(__file__))
        self.crib_discards = np.load(os.path.join(cwd_name, "crib_discards.npy"))
        self.opponent_discards = np.load(os.path.join(cwd_name, "opponent_discards.npy"))

    def discard_crib_pretty(self, hand, is_dealer, my_score, opponents_score):
        print("Considering hands from {} {} {} {} {} {}".format(*[card_to_string(h) for h in hand]))
        discards = self.bfs(hand, is_dealer, my_score, opponents_score)
        for choice in discards:
            new_hand = deepcopy(hand)
            new_hand.remove(hand[choice[1]])
            new_hand.remove(hand[choice[2]])
            new_hand_string = [card_to_string(h) for h in new_hand]
            discard_val = sorted([hand[choice[1]][0], hand[choice[2]][0]])
            if is_dealer:
                discard_points = self.crib_discards[discard_val[0]-1, discard_val[1]-1]
            else:
                discard_points = -self.opponent_discards[discard_val[0]-1, discard_val[1]-1]
            new_hand_string.append(-choice[0] - discard_points)
            new_hand_string.append(discard_points)
            new_hand_string.append(-choice[0])
            print("{:>3} {:>3} {:>3} {:>3}    Mean score: {:5.2f}  Crib: {:5.2f}  Total: {:6.2f}".format(*new_hand_string))

    def discard_crib(self, hand, is_dealer, my_score, opponents_score):
        """
        Thy
        :param hand: list of 6 cards, each card is a tuple of value and suit
        :param is_dealer: whether the player is the dealer
        :return: the tuple of 2 cards to discard
        """
        possible_choices = self.bfs(hand, is_dealer, my_score, opponents_score)
        highest_score_hand = possible_choices[0]
        (points, first_index, second_index) = highest_score_hand
        return hand[first_index], hand[second_index]

    def bfs(self, hand, is_dealer, my_score, opponents_score):
        """
        Thy
        Generates a priority queue based on 4-card hand scores with different permutations of 2 discarded cards
        :param hand: 6-card hand
        :return: a priority queue of 4-card hand based on their scores
        """
        priorityq = []
        possible_cut_cards = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1),
                              (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2), (13, 2),
                              (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3), (13, 3),
                              (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4), (13, 4)]

        for i in range(len(hand)):

            first_removed = hand[i]
            for j in range(i+1, len(hand)):
                copyhand = deepcopy(hand)
                second_removed = hand[j]
                copyhand.remove(first_removed)
                copyhand.remove(second_removed)

                points = []

                for cut_card in possible_cut_cards:
                    if not cut_card in hand:
                        points.append(score_hand(copyhand, cut_card))

                points = np.mean(points)
                discards = sorted([first_removed[0], second_removed[0]])
                
                if is_dealer:
                    points += self.crib_discards[discards[0]-1, discards[1]-1]
                else:
                    points -= self.opponent_discards[discards[0]-1, discards[1]-1]
                
                heapq.heappush(priorityq, (-points, i, j))

        return priorityq

    def pegging_move(self, hand, sequence, current_sum):
        """
        Greedy pegging moves
        """
        # exclude cards that can't be played
        
        cards = [h for h in hand if current_sum + peg_val(h) <= 31]
        
        if len(cards) == 0:
            return None
        
        def check_run(target_length, cards, played):
            if len(played) >= target_length - 1:
                for c in cards:
                    p = [pl[0] for pl in played]
                    p.append(c[0])
                    p = sorted(p[-target_length:])
                    is_run = True
                    for j in range(len(p)-1):
                        if not p[j] == p[j + 1] - 1:
                            is_run = False
                    if is_run:
                        return c
                return None
            else:
                return None
        
        # 4 of a kind
        
        if len(sequence) >= 3:
            if sequence[-1][0] == sequence[-2][0] and sequence[-1][0] == sequence[-3][0] and sequence[-1][0] in [c[0] for c in cards]:
                for c in cards:
                    if c[0] == sequence[-1][0]:
                        return c
                        
        # Run of 7

        c = check_run(7, cards, sequence)
        if not c is None:
            return c
                    
        # Run of 6
        
        c = check_run(6, cards, sequence)
        if not c is None:
            return c
                    
        # 3 of a kind
        
        if len(sequence) >= 2:
            if sequence[-1][0] == sequence[-2][0] and sequence[-1][0] in [c[0] for c in cards]:
                for c in cards:
                    if c[0] == sequence[-1][0]:
                        return c
        
        # run of 5
        
        c = check_run(5, cards, sequence)
        if not c is None:
            return c
            
        # run of 4
        
        c = check_run(4, cards, sequence)
        if not c is None:
            return c
            
        # run of 3
        
        c = check_run(3, cards, sequence)
        if not c is None:
            return c
            
        # 31
        
        for c in cards:
            if current_sum + peg_val(c) == 31:
                return c
                
        # 15
        
        for c in cards:
            if current_sum + peg_val(c) == 15:
                return c
                
        # Else play card that maximizes the running score
        
        cmax = cards[0]
        for c in cards:
            if peg_val(c) > peg_val(cmax):
                cmax = c
                
        return cmax
        
if __name__ == "__main__":
    agent = MCAgent()
    
    agent.discard_crib_pretty([(4, 1), (5, 2), (10, 0), (11, 1), (9, 0), (4, 2)], True, 0, 0)
    
    agent.discard_crib_pretty([(4, 1), (5, 2), (10, 0), (11, 1), (9, 0), (4, 2)], False, 0, 0)