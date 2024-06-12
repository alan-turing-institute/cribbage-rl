from unittest import TestCase, main

import numpy as np

from new_environment import get_greedy_action, CribbageEnv

class TestNewEnvironment(TestCase):
    hand_one = [
        (1, "S"),
        (2, "S"),
        (3, "S"),
        (4, "S"),
        (5, "S"),
        (6, "S"),
    ]

    def new_environment_test(self):
        actual_discard, actual_keep, actual_score = get_greedy_action((
            np.array([0, 0], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([2, 0], dtype=np.int64),
            np.array([3, 0], dtype=np.int64),
            np.array([4, 0], dtype=np.int64),
            np.array([5, 0], dtype=np.int64),
            
        ))
        expected_discard, expected_keep, expected_score = (
            (
                np.array([0, 0], dtype=np.int64),
                np.array([1, 0], dtype=np.int64),
            ),
            (
                np.array([2, 0], dtype=np.int64),
                np.array([3, 0], dtype=np.int64),
                np.array([4, 0], dtype=np.int64),
                np.array([5, 0], dtype=np.int64),
            ),
            10
        )
        
        self.assertEqual(actual_score, expected_score)
        self.assertTupleEqual(actual_keep, expected_keep)
        self.assertTupleEqual(actual_discard, expected_discard)

    def test_discard(self):
        environment = CribbageEnv()
        environment.reset()
        environment.current_hand = self.hand_one
        environment.starter_card = (10, "C")
        actual = environment.discard(0)
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
    test_new_environment = TestNewEnvironment()
    test_new_environment.test_discard()