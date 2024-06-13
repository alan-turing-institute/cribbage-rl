from itertools import combinations

from cribbage_scorer import cribbage_scorer
from tqdm import tqdm

POSSIBLE_ACTIONS = list(combinations(list(range(6)), 2))


def run(
    hand_cards=[
        (3, 2),
        (7, 1),
        (8, 3),
        (10, 1),
        (12, 2),
        (7, 0),
    ],
    is_dealer=False,
):
    crib_factor = 1 if is_dealer else -1
    action_scores = {}
    total_counts = (52 - 6) * (52 - 7) * (52 - 8)
    pbar = tqdm(total=len(POSSIBLE_ACTIONS) * total_counts)
    for drop_choice in POSSIBLE_ACTIONS:
        hand_choices = [
            card for i, card in enumerate(hand_cards) if i not in drop_choice
        ]
        crib_choices = [card for i, card in enumerate(hand_cards) if i in drop_choice]
        running_score = 0
        for starter_card in range(52):
            starter_card = (starter_card % 13, starter_card // 13)
            if starter_card in hand_cards:
                continue
            for opponent_1 in range(52):
                opponent_1 = (opponent_1 % 13, opponent_1 // 13)
                if opponent_1 in hand_cards or opponent_1 == starter_card:
                    continue
                for opponent_2 in range(52):
                    opponent_2 = (opponent_2 % 13, opponent_2 // 13)
                    if (
                        opponent_2 in hand_cards
                        or opponent_2 == starter_card
                        or opponent_2 == opponent_1
                    ):
                        continue
                    pbar.update(1)
                    running_score += cribbage_scorer.show_calc_score(
                        starter_card,
                        hand_choices,
                        crib=False,
                    )[0]
                    running_score += (
                        crib_factor
                        * cribbage_scorer.show_calc_score(
                            starter_card,
                            [
                                opponent_1,
                                opponent_2,
                                crib_choices[0],
                                crib_choices[1],
                            ],
                            crib=True,
                        )[0]
                    )
        running_score /= total_counts
        action_scores[drop_choice] = running_score
    pbar.close()
    print(action_scores)
    best_choice = max(action_scores, key=action_scores.get)
    print(best_choice, action_scores[best_choice])


if __name__ == "__main__":
    run()
