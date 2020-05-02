import numpy as np


def divisor_method(votes: np.ndarray, seats: int, step=1) -> np.ndarray:
    assert seats > 0, "at least one seat required"
    assert votes.sum() >= seats, "the number of votes should be greater or equal than the number of seats"
    outcome = np.zeros_like(votes)
    votes_c = np.copy(votes)
    divisors = np.ones_like(votes)
    # step = 2 if saint_lague else 1

    for _ in range(seats):
        seat_taker = votes_c.argmax()
        outcome[seat_taker] += 1
        divisors[seat_taker] += step
        votes_c[seat_taker] = votes[seat_taker] / divisors[seat_taker]

    return outcome

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

