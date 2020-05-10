import numpy as np
import numpy.ma as ma


def SNTV_scores(ballots: np.ndarray, seats: int, electoral_threshold=0, percentage=False):
    """

    :param ballots: (n x m) matrix that contains for each voter i the index of the party ranked at position j ballots[i,j]
    :param seats: number of seats available
    :param electoral_threshold: the electoral threshold
    :param percentage: if the electoral threshold is in fraction or in number of voters
    :return: (votes for running parties, votes for all, mask for running parties)
    """
    if percentage:
        electoral_threshold = np.floor(electoral_threshold * ballots.shape[0])

    # counts the first choices in the ballot
    plurality_scores = np.bincount(ballots[:, 0], minlength=m)

    # parties whose score passes the threshold
    passed_parties = plurality_scores >= electoral_threshold

    return plurality_scores * passed_parties, plurality_scores, passed_parties


def STV_scores(ballots: np.ndarray, seats: int, electoral_threshold=0, percentage=False):
    """

    :param ballots: (n x m) matrix that contains for each voter i the index of the party ranked at position j ballots[i,j]
    :param seats: number of seats available
    :param electoral_threshold: the electoral threshold
    :param percentage: if the electoral threshold is in fraction or in number of voters
    :return:
    """
    n, m = ballots.shape
    if percentage:
        electoral_threshold = np.floor(electoral_threshold * n)
    alive_parties = np.full(m, True)

    while True:
        # count the occurrence of each party in top choice
        plurality_scores = np.bincount(ballots[:, 0], minlength=m)[:m]

        # picks the party with the lowest number of votes
        # among all the ones which are still running
        losing_party = ma.array(plurality_scores, mask=~alive_parties) \
            .argmin(fill_value=n + 1)

        if plurality_scores[losing_party] >= electoral_threshold:
            # the remaining party with the lowest
            # amount of votes passes the quota
            break
        else:
            # the party does not pass the quota
            alive_parties[losing_party] = 0
            voters_to_change = np.argwhere(ballots[:, 0] == losing_party)
            # mark where in the ballots the dead party is voted
            # sets it to m, unreachable index
            ballots = np.where(ballots == losing_party, m, ballots)

            for i in voters_to_change:
                v = i[0]
                # counter = 0
                shift = np.nonzero(ballots[v] < m)[0][0]
                ballots[v] = np.roll(ballots[v], -shift)
                # while ballots[v, 0] == m:
                #     # "roll" the ballot to the left until
                #     # the top-choice party is not a dead party
                #
                #     ballots[v] = np.roll(ballots[v], -1)
                #     counter += 1
                #     if counter > m:
                #         raise RuntimeError
                # assert shift == counter

    return plurality_scores * alive_parties, plurality_scores, alive_parties


def alpha_STV_scores(ballots: np.ndarray, seats: int, alpha=0.8, electoral_threshold=0, percentage=False):
    """

    :param ballots: (n x m) matrix that contains for each voter i the index of the party ranked at position j ballots[i,j]
    :param seats: number of seats available
    :param alpha: decay factor for vote value
    :param electoral_threshold: the electoral threshold
    :param percentage: if the electoral threshold is in fraction or in number of voters
    :return:
    """
    n, m = ballots.shape
    if percentage:
        electoral_threshold = np.floor(electoral_threshold * n)
    passed_parties = np.full(m, True)
    vote_values = np.ones(n)

    while True:
        # count the occurrence of each party in top choice
        plurality_scores = np.bincount(ballots[:, 0], weights=vote_values, minlength=m)[:m]

        # picks the party with the lowest number of votes
        # among all the ones which are still running
        losing_party = ma.array(plurality_scores, mask=~passed_parties) \
            .argmin(fill_value=n + 1)

        if plurality_scores[losing_party] >= electoral_threshold:
            # the remaining party with the lowest
            # amount of votes passes the quota
            break
        else:
            # the party does not pass the quota
            passed_parties[losing_party] = 0
            voters_to_change = np.argwhere(ballots[:, 0] == losing_party)
            # mark where in the ballots the dead party is voted
            # sets it to m, unreachable index
            ballots = np.where(ballots == losing_party, m, ballots)

            for i in voters_to_change:
                v = i[0]
                counter = 0
                # shift = np.nonzero(ballots[v] < m)[0][0]
                # vote_values[v] *= alpha**shift
                # ballots[v] = np.roll(ballots[v], -shift)
                while ballots[v, 0] == m:
                    vote_values[v] *= alpha
                    # "roll" the ballot to the left until
                    # the top-choice party is not a dead party
                    ballots[v] = np.roll(ballots[v], -1)
                    counter += 1
                    if counter > m:
                        raise RuntimeError

    return plurality_scores * passed_parties, plurality_scores, passed_parties


if __name__ == '__main__':
    n = 30
    m = 5
    e = 0.30
    random = np.random.default_rng()
    ballots = np.stack([random.permutation(m) for _ in range(n)])
    from pprint import pprint

    pprint(ballots)
    pprint(SNTV_scores(ballots, 3, electoral_threshold=e, percentage=True))
    pprint(STV_scores(ballots, 3, electoral_threshold=e, percentage=True))
    pprint(alpha_STV_scores(ballots, 3, alpha=0, electoral_threshold=e, percentage=True))
    pprint(alpha_STV_scores(ballots, 3, alpha=1, electoral_threshold=e, percentage=True))
    pprint(alpha_STV_scores(ballots, 3, electoral_threshold=e, percentage=True))
