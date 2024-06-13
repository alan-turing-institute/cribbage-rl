import unittest

import main


class TestMain(unittest.TestCase):
    def test_main(self) -> None:
        self.assertEqual(main.main(), 0)

    def test_get_optimal_strategy(self) -> None:

        self.assertListEqual(
            main.evaluate_strategies(
                (0, 0),
                10,
            ),
            [1.0, 1.0],
        )
