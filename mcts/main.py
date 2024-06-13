from functools import cache
import random
from typing import Iterable, Callable, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


# Each strategy should return a score for the current round.
strategies: Optional[Iterable[Callable[[], int]]] = None


def conservative() -> int:
    scores = 3, 4, 5
    return random.choices(scores, weights=[1 for _ in scores])[0]


def agressive() -> int:
    scores = 1, 2, 3, 4, 5, 6, 7
    return random.choices(scores, weights=[1 for _ in scores])[0]


@cache
def evaluate_strategies(
    scores: tuple[int, int], target: int, samples: int
) -> list[float]:
    """Evaluate the win rate of each strategy in the global iterable of strategies."""
    assert strategies is not None
    win_rates = []
    for strategy in strategies:
        # Take an average
        wins = 0.0
        for _ in range(samples):
            my_score = strategy()
            if my_score + scores[0] >= target:
                wins += 1.0
            else:
                # Another round, this time the opponent goes first
                wins += 1 - max(
                    evaluate_strategies(
                        (scores[1], scores[0] + my_score), target, samples
                    )
                )
        win_rates.append(wins / samples)
    return win_rates


def main() -> None:
    """Evaluate the best strategy for each possible state."""
    global strategies
    max_score = 120
    data: npt.NDArray[np.int_] = np.ndarray(shape=(max_score, max_score), dtype=int)
    strategies = [agressive, conservative]
    for x in range(max_score):
        for y in range(max_score):
            data[x, y] = np.argmax(evaluate_strategies((x, y), 121, 100))
    plt.imshow(
        data,
        cmap="hot",
        interpolation="nearest",
        origin="lower",
    )
    plt.savefig("./cribbage.png")


if __name__ == "__main__":
    main()
