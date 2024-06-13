from functools import cache
from typing import Iterable, Callable, Optional


strategies: Optional[Iterable[Callable[[], int]]] = None


@cache
def evaluate_strategies(scores: tuple[int, int], target: int, samples) -> list[float]:
    assert strategies is not None
    win_rates = []
    for strategy in strategies:
        wins = 0.0
        for _ in range(samples):
            my_score = strategy()
            if my_score + scores[0] >= target:
                wins += 1.0
            else:
                # another round, this time the opponent goes first
                wins += 1 - max(
                    evaluate_strategies(
                        (scores[1], scores[0] + my_score), target, samples
                    )
                )
        win_rates.append(wins / samples)
    return win_rates


def main() -> int:
    return 0
