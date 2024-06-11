"""Cribbage score conditions used during and after rounds."""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import combinations


class ScoreCondition(metaclass=ABCMeta):
    """Abstract Base Class"""

    def __init__(self):
        pass

    @abstractmethod
    def check(self, hand):
        raise NotImplementedError


class HasPairTripleQuad_DuringPlay(ScoreCondition):
    def check(self, cards):
        description = None
        pair_rank = ""
        same, score = 0, 0
        if len(cards) > 1:
            last = cards[-4:][::-1]
            while same == 0 and last:
                if all(card.rank["name"] == last[0].rank["name"] for card in last):
                    same = len(last)
                    pair_rank = last[0].rank["symbol"]
                last.pop()
            if same == 2:
                score = 2
                description = "Pair (%s)" % pair_rank
            elif same == 3:
                score = 6
                description = "Pair Royal (%s)" % pair_rank
            elif same == 4:
                score = 12
                description = "Double Pair Royal (%s)" % pair_rank
        return score, description


class HasPairTripleQuad_InHand(ScoreCondition):
    def check(self, cards):
        pair_rank = []
        triple_rank = []
        quad_rank = []
        card_ranks = [card.rank["name"] for card in cards]
        for card_rank in card_ranks:
            if card_ranks.count(card_rank) == 2:
                pair_rank.append(card_rank)
            elif card_ranks.count(card_rank) == 3:
                triple_rank.append(card_rank)
            elif card_ranks.count(card_rank) == 4:
                quad_rank.append(card_rank)
        if len(pair_rank) == 4:
            second_pair = pair_rank[1] if pair_rank[0] != pair_rank[1] else pair_rank[2]
            return 4, "2 pairs of %s and %s" % (pair_rank[0], second_pair)
        elif len(pair_rank) == 2 and len(triple_rank) == 3:
            return 8, "Pair of %s and Triple of %s" % pair_rank[0], triple_rank[0]
        elif len(pair_rank) == 2:
            return 2, "Pair of %s" % pair_rank[0]
        elif len(triple_rank) == 3:
            return 6, "Triple of %s" % triple_rank[0]
        elif len(quad_rank) == 4:
            return 12, "Quad of %s" % quad_rank[0]
        else:
            return 0, ""


class ExactlyEqualsN(ScoreCondition):

    def __init__(self, n):
        self.n = n
        super().__init__()

    def check(self, cards):
        value = sum(i.get_value() for i in cards)
        score = 2 if value == self.n else 0
        description = "%d count" % self.n if score else ""
        return score, description


class HasStraight_InHand(ScoreCondition):

    @staticmethod
    def _enumerate_straights(cards):
        potential_straights = []
        straights = []
        straights_deduped = []
        if cards:
            for i in range(3, len(cards) + 1):
                potential_straights += list(combinations(cards, i))
            for p in potential_straights:
                rank_set = set([card.rank["rank"] for card in p])
                if (max(rank_set) - min(rank_set) + 1) == len(p) == len(rank_set):
                    straights.append(set(p))
            for s in straights:
                subset = False
                for o in straights:
                    if s.issubset(o) and s is not o:
                        subset = True
                if not subset:
                    straights_deduped.append(s)
        return straights_deduped

    @classmethod
    def check(cls, cards):
        description = ""
        points = 0
        straights = cls._enumerate_straights(cards)
        for s in straights:
            assert len(s) >= 3, "Straights must be 3 or more cards."
            description += "%d-card straight " % len(s)
            points += len(s)
        return points, description


class HasStraight_DuringPlay(ScoreCondition):

    @staticmethod
    def _is_straight(cards):
        rank_set = set([card.rank["rank"] for card in cards])
        return (
            ((max(rank_set) - min(rank_set) + 1) == len(cards) == len(rank_set))
            if len(cards) > 2
            else False
        )

    @classmethod
    def check(cls, cards):
        description = ""
        card_set = cards[:]
        while card_set:
            if cls._is_straight(card_set):
                description = "%d-card straight" % len(card_set)
                return len(card_set), description
            card_set.pop(0)
        return 0, description


class CountCombinationsEqualToN(ScoreCondition):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def check(self, cards):
        n_counts, score = 0, 0
        cmb_list = []
        card_values = [card.get_value() for card in cards]
        for i in range(len(card_values)):
            cmb_list += list(combinations(card_values, i + 1))
        for i in cmb_list:
            n_counts += 1 if sum(i) == self.n else 0
        description = "%d unique %d-counts" % (n_counts, self.n) if n_counts else ""
        score = n_counts * 2
        return score, description


class HasFlushHand(ScoreCondition):
    """Check for a 4 or 5 card flush in a hand.
    The last card must be the starter card.
    """

    def check(self, cards):
        card_suits = [card.get_suit() for card in cards]
        if card_suits.count(cards[0].get_suit()) == 5:
            return 5, "5-card flush"
        elif card_suits[:4].count(cards[0].get_suit()) == 4:
            return 4, "4-card flush"
        else:
            return 0, ""


class HasFlushCrib(ScoreCondition):
    def check(self, cards):
        card_suits = [card.get_suit() for card in cards]
        if card_suits.count(cards[0].get_suit()) == 5:
            return 5, "5-card flush"
        else:
            return 0, ""
