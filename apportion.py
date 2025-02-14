import numpy as np


def divisor_method(votes: np.ndarray, seats: int, step: int) -> np.ndarray:
    """

    :param votes: one-dimensional array of shape (m) where m is the number of parties
    :param seats: the total number of seats available
    :param step: the increasing in the divisors
    :return:
    """
    assert seats > 0, "at least one seat required"
    assert votes.sum() >= seats, "the number of votes should be greater or equal than the number of seats"
    outcome = np.zeros_like(votes)
    votes_c = np.copy(votes)
    divisors = np.ones_like(votes)

    for _ in range(seats):
        seat_taker = votes_c.argmax()
        outcome[seat_taker] += 1
        divisors[seat_taker] += step
        votes_c[seat_taker] = votes[seat_taker] / divisors[seat_taker]

    return outcome


def dhondt(votes: np.ndarray, seats: int) -> np.ndarray:
    return divisor_method(votes, seats, 1)


def saint_lague(votes: np.ndarray, seats: int) -> np.ndarray:
    return divisor_method(votes, seats, 2)


def largest_remainder(votes: np.ndarray, seats: int) -> np.ndarray:
    assert seats > 0, "at least one seat required"
    assert votes.sum() >= seats, "the number of votes should be greater or equal than the number of seats"
    quota = votes.sum() / seats
    outcome, remainders = np.divmod(votes, quota)
    outcome = outcome.astype(int)
    remaining_seats = int(seats - outcome.sum())
    to_add = np.argpartition(-remainders, remaining_seats)[:remaining_seats]
    outcome[to_add] += 1

    return outcome

def tryall(votes: np.ndarray, seats: int):
    print(f"D'Hondt\t\t\t{dhondt(votes,seats)}")
    print(f"SaintLague\t\t{saint_lague(votes,seats)}")
    print(f"LargestRemainder\t{largest_remainder(votes,seats)}")