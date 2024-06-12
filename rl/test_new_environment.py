from unittest import TestCase, main

import numpy as np

from new_environment import CribbageEnv

class TestNewEnvironment(TestCase):
    hand_one = [
        (1, "S"),
        (2, "S"),
        (3, "S"),
        (4, "S"),
        (5, "S"),
        (6, "S"),
    ]
    def setUp(self):
        environment = CribbageEnv()
        environment.reset()
        environment.current_hand = self.hand_one.copy()
        environment.dealt_hand = environment.current_hand.copy()
        environment.starter_card = (10, "C")
        self.env = environment


    def test_get_greedy_hand(self):
        actual_original, actual_kept, actual_score = self.env.get_greedy_hand()
        
        expected_original, expected_kept, expected_score = (
            [
                (1, "S"),
                (2, "S"),
                (3, "S"),
                (4, "S"),
                (5, "S"),
                (6, "S"),
            ],
            [
                (1, "S"),
                (4, "S"),
                (5, "S"),
                (6, "S"),

            ],
            13
        )

        self.assertEqual(actual_score, expected_score)
        self.assertListEqual(actual_kept, expected_kept)
        self.assertListEqual(actual_original, expected_original)

    def test_discard(self):
        # 0 is the index of the first potential action.
        actual = self.env.discard(0)
        expected = (
            [
                (1, "S"),
                (2, "S"),
                (3, "S"),
                (4, "S"),
                (5, "S"),
                (6, "S"),
            ],
            [
                (3, "S"),
                (4, "S"),
                (5, "S"),
                (6, "S"),

            ],
            12
        )
        self.assertEqual(actual[2], expected[2])
        self.assertListEqual(actual[0], expected[0])
        self.assertListEqual(actual[1], expected[1])


if __name__ == '__main__':
    main()
