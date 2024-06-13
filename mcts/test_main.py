import unittest
from random import choices

import main


def strategy1() -> int:
    return 2


def strategy2() -> int:
    return 1


def strategy3() -> int:
    # return 4 with 50% probability and 2 with 50% probability
    return choices([4, 2], weights=[0.5, 0.5])[0]


def strategy4() -> int:
    return 3


class TestMain(unittest.TestCase):
    def test_eval_one(self) -> None:
        main.evaluate_strategies.cache_clear()
        main.strategies = [strategy1, strategy2]
        self.assertListEqual(
            main.evaluate_strategies((0, 0), 10, 10),
            [1.0, 0.0],
        )

    def test_eval_two(self) -> None:
        main.evaluate_strategies.cache_clear()
        main.strategies = [strategy3, strategy4]
        rates = main.evaluate_strategies((0, 0), 10, 100)
        self.assertTrue(rates[1] + 0.1 > rates[0] > rates[1] - 0.1)

    def test_eval_three(self) -> None:
        main.evaluate_strategies.cache_clear()
        main.strategies = [strategy4, strategy4]
        rates = main.evaluate_strategies((9, 9), 10, 100)
        self.assertTrue(rates[0] == rates[1] == 1.0)

    def test_eval_cribbage(self) -> None:
        main.evaluate_strategies.cache_clear()
        main.strategies = [main.agressive, main.conservative]

        print("first move")
        rates = main.evaluate_strategies((0, 0), 121, 400)
        print("aggressive, conservative", rates)

        print("last move 1")
        rates = main.evaluate_strategies((118, 118), 121, 400)
        print("aggressive, conservative", rates)

        print("last move 2")
        rates = main.evaluate_strategies((114, 114), 121, 400)
        print("aggressive, conservative", rates)

        print("behind 1")
        rates = main.evaluate_strategies((0, 9), 121, 400)
        print("aggressive, conservative", rates)

        print("behind 2")
        rates = main.evaluate_strategies((90, 97), 121, 400)
        print("aggressive, conservative", rates)

        print("ahead")
        rates = main.evaluate_strategies((9, 0), 121, 400)
        print("aggressive, conservative", rates)
