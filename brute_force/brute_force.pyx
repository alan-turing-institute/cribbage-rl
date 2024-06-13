from itertools import combinations

from cython.parallel import prange

from tqdm import tqdm

from scorer import score_hand

POSSIBLE_ACTIONS = list(combinations(list(range(6)), 2))


def get_best_action(hand_cards, is_dealer):
    crib_factor = 1 if is_dealer else -1
    action_scores = {}
    total_counts = (52 - 6) * (52 - 7) * (52 - 8)
    pbar = tqdm(total=len(POSSIBLE_ACTIONS) * (52 - 6))
    card_deck = [(x % 13, x // 13) for x in range(52)]
    for i in prange(len(POSSIBLE_ACTIONS))
        drop_choice = POSSIBLE_ACTIONS[i]
        hand_choices = [
            card for i, card in enumerate(hand_cards) if i not in drop_choice
        ]
        crib_choices = [card for i, card in enumerate(hand_cards) if i in drop_choice]
        running_score = 0
        for starter_card in card_deck:
            if starter_card in hand_cards:
                continue
            pbar.update(1)
            for opponent_1 in card_deck:
                if opponent_1 in hand_cards or opponent_1 == starter_card:
                    continue
                for opponent_2 in card_deck:
                    if (
                        opponent_2 in hand_cards
                        or opponent_2 == starter_card
                        or opponent_2 == opponent_1
                    ):
                        continue
                    running_score += score_hand(
                        hand_choices,
                        starter_card,
                        is_crib=False,
                    )
                    running_score += crib_factor * score_hand(
                        [
                            opponent_1,
                            opponent_2,
                            crib_choices[0],
                            crib_choices[1],
                        ],
                        starter_card,
                        is_crib=True,
                    )
        running_score /= total_counts
        action_scores[drop_choice] = running_score
    pbar.close()
    best_choice = max(action_scores, key=action_scores.get)
    return best_choice, action_scores[best_choice]
