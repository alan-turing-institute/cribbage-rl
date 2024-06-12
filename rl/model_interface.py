from itertools import combinations

import click
import numpy as np
from stable_baselines3 import PPO

ALL_SUITS: list[str] = ["D", "S", "C", "H"]
POSSIBLE_ACTIONS = list(combinations(list(range(6)), 2))


@click.command()
@click.argument("model_file")
def run(model_file):
    click.echo(f"Loading model {model_file}")
    model = PPO.load(model_file)

    card_ranks = []
    card_suits = []
    is_dealer = False

    is_dealer = click.prompt("Are you dealer? ", type=bool)

    for i in range(1, 7):
        card_suit = click.prompt(f"Card {i} suit: ", type=str)
        if card_suit not in ALL_SUITS:
            click.echo(f"Invalid suit: {card_suit}, please use one of {ALL_SUITS}")
            raise click.Abort()
        card_rank = click.prompt(f"Card {i} rank: ", type=int)
        if card_rank < 1 or card_rank > 13:
            click.echo(
                f"Invalid rank: {card_rank}, please use a number between 1 and 13"
            )
            raise click.Abort()

        card_suits.append(card_suit)
        card_ranks.append(card_rank)

    suit_to_number = {suit: i for i, suit in enumerate(ALL_SUITS)}

    observation = {}

    for n, (card_rank, card_suit) in enumerate(zip(card_ranks, card_suits)):
        observation[f"card_{n}"] = np.array([card_rank - 1, suit_to_number[card_suit]])

    observation["is_dealer"] = np.array([is_dealer])

    action, _state = model.predict(observation, deterministic=True)

    cards_to_drop = POSSIBLE_ACTIONS[action[0]]

    click.echo(
        f"Drop card {cards_to_drop[0] + 1}, "
        f" {card_suits[cards_to_drop[0]]} {card_ranks[cards_to_drop[0]]}"
    )
    click.echo(
        f"Drop card {cards_to_drop[1] + 1}, "
        f" {card_suits[cards_to_drop[1]]} {card_ranks[cards_to_drop[1]]}"
    )


if __name__ == "__main__":
    run()
