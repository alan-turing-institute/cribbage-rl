from itertools import combinations

import click

from brute_force import get_best_action

ALL_SUITS: list[str] = ["D", "S", "C", "H"]
POSSIBLE_ACTIONS = list(combinations(list(range(6)), 2))


@click.command()
def run():
    keep_playing = True

    while keep_playing:

        card_ranks = []
        card_suits = []
        is_dealer = False

        is_dealer = click.prompt("Are you dealer? ", type=bool)

        for i in range(1, 7):
            card_choice = click.prompt(f"Card {i} suit then rank: ", type=str)
            if len(card_choice) < 2 or len(card_choice) > 4:
                click.echo(
                    f"Invalid suit and rank: {card_choice}, please use format 'D 1'"
                )
                raise click.Abort()
            card_suit = card_choice[0]
            card_rank = int(card_choice[1:])
            if card_suit not in ALL_SUITS:
                click.echo(f"Invalid suit: {card_suit}, please use one of {ALL_SUITS}")
                raise click.Abort()
            if card_rank < 1 or card_rank > 13:
                click.echo(
                    f"Invalid rank: {card_rank}, please use a number between 1 and 13"
                )
                raise click.Abort()

            card_suits.append(card_suit)
            card_ranks.append(card_rank)

        suit_to_number = {suit: i for i, suit in enumerate(ALL_SUITS)}

        hand_cards = [
            (card_rank - 1, suit_to_number[card_suit])
            for card_rank, card_suit in zip(card_ranks, card_suits)
        ]

        best_choice, _ = get_best_action(hand_cards, is_dealer)

        cards_to_drop = best_choice

        click.echo(
            f"Drop card {cards_to_drop[0] + 1}, "
            f" {card_suits[cards_to_drop[0]]} {card_ranks[cards_to_drop[0]]}"
        )
        click.echo(
            f"Drop card {cards_to_drop[1] + 1}, "
            f" {card_suits[cards_to_drop[1]]} {card_ranks[cards_to_drop[1]]}"
        )

        keep_playing = click.prompt("Do you want to keep playing?", type=bool)

    click.echo("Thank you for a lovely game of cribbage!")


if __name__ == "__main__":
    run()
