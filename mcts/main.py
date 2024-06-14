from functools import cache
import random
from pathlib import Path
from typing import Iterable, Callable, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


# Each strategy should return a score for the current round.
strategies: Optional[Iterable[Callable[[], int]]] = None


def conservative() -> int:
    scores = 9,
    return random.choices(scores, weights=[1 for _ in scores])[0]


def agressive() -> int:
    scores = 5, 11
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
    max_score = 121
    iterations = 100

    filename = f"./cribbage_{max_score}_{iterations}.npy"
    if Path(filename).exists():
        print("loading")
        data = np.load(filename)
    else:
        data: npt.NDArray[np.int_] = np.ndarray(shape=(max_score-1, max_score-1, 3), dtype=np.float32)
        strategies = [agressive, conservative]
        for x in range(max_score-1):
            for y in range(max_score-1):
                prob_agg, prob_cons = evaluate_strategies((x, y), 121, 100)
                # data[x, y] = 0 if prob_agg > prob_cons else 1
                data[x, y] = prob_agg, prob_cons, 0  # RGB
                # contour_data[x, y] =  0.5 + (0.5 * (prob_agg - prob_cons)) / (prob_agg + prob_cons + 1e-12)
                # contour_data[x, y] = 0.5 + (0.5 * (prob_agg - prob_cons)) / (prob_agg + prob_cons + 1e-6)
                # colour_data[x, y] = prob_agg, 0, prob_cons

        np.save("./cribbage_121_100.npy", data)


    # contour_data: npt.NDArray[np.float_] = np.ndarray(shape=(max_score - 1, max_score - 1), dtype=np.float32)
    # colour_data: npt.NDArray[np.float_] = np.ndarray(shape=(max_score - 1, max_score - 1, 3), dtype=np.float32)
    # exit(0)
    # print(contour_data[10:20, 50])
    plt.imshow(
        data,
        # cmap="hot",
        interpolation="nearest",
        origin="lower",
    )
    # plt.contourf(
    #     contour_data,
    #     cmap="RdYlBu",
    #     levels=30,
    #     origin="lower",
    # )
    plt.savefig("./cribbage_1.png")


if __name__ == "__main__":
    main()
