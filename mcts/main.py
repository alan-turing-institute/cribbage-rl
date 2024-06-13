from functools import cache
from typing import Iterable, Callable

SAMPLES = 10


def strategy1() -> tuple[int, int]:
    return 1, 0


def strategy2() -> tuple[int, int]:
    return 0, 1


strategies: Iterable[Callable[[], tuple[int, int]]] = [
    strategy1,
    strategy2,
]


@cache
def evaluate_strategies(scores: tuple[int, int], target: int) -> list[float]:
    win_rates = []
    for strategy in strategies:
        wins = 0.0
        draws = 0.0
        for _ in range(SAMPLES):
            my_score, their_score = strategy()
            new_scores = scores[0] + my_score, scores[1] + their_score
            if new_scores[0] > target:
                # win or draw
                if new_scores[1] > target:
                    # draw
                    draws += 1.0
                else:
                    # win
                    wins += 1.0
            elif new_scores[1] > target:
                # loss
                pass
            else:
                # another round
                wins += max(evaluate_strategies(new_scores, target))
        win_rates.append(wins / SAMPLES)
    return win_rates


def main() -> int:
    return 0
