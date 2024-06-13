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
    def test_main(self) -> None:
        self.assertEqual(main.main(), 0)

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
