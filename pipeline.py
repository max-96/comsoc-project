from math import factorial

import apportion
import utils
import voting
from voting import *
from apportion import *
from utils import PreferenceCreator
from tabulate import tabulate
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
n = 30
m = 8
e = 0.1
seats = 3


def get_poll(preferences: np.ndarray, probability: float):
    assert 0. <= probability <= 1., "{:.4f} is not a probability".format(probability)
    n_voters = preferences.shape[0]
    n_sample = round(n_voters * probability)
    indices = np.random.default_rng().permutation(n_voters)[:n_sample]

    samples_prefs = preferences[indices, 0]
    results = np.bincount(samples_prefs, minlength=preferences.shape[1]) / n_sample
    return results


def get_first_choice_dist(preferences: np.ndarray):
    return np.bincount(preferences[:, 0], minlength=preferences.shape[1]) / preferences.shape[0]


def main():
    n_tests = 10
    m = 6
    n = 100
    seats = 5
    electoral_threshold = 0.05
    # political_spectrum = np.array([5, 4, 6, 3, 7, 2, 8, 1, 9, 0])

    props = []
    progressbar = tqdm(total=factorial(m))
    for p in permutations(range(m)):
        progressbar.update()
        political_spectrum = np.array(p)

        a = lalala(electoral_threshold, m, n, n_tests, political_spectrum, seats)
        props.append(a)
    progressbar.close()
    mean_stv_prop, mean_sntv_prop = list(zip(*props))
    a = np.array(mean_stv_prop)
    b = np.array(mean_sntv_prop)
    print(a, b)
    plt.plot(a, label='stv')
    plt.plot(b, label='sntv')
    plt.legend()
    plt.show()


def lalala(electoral_threshold, m, n, n_tests, political_spectrum, seats):
    stv_propor = []
    sntv_propor = []
    for test_nr in range(n_tests):
        true_preferences = PreferenceCreator(n, m, political_spectrum).create_preferences()

        # STV outcome
        stv_scores, *_ = voting.STV_scores(true_preferences, electoral_threshold, percentage=True)
        stv_outcome = apportion.largest_remainder(stv_scores, seats)

        # SNTV outcome
        sntv_scores, *_ = voting.SNTV_scores(true_preferences, electoral_threshold, percentage=True)
        sntv_outcome = apportion.largest_remainder(sntv_scores, seats)

        first_distribution = get_first_choice_dist(true_preferences)

        # Proportionality
        stv_propor.append(utils.kl_divergence(stv_outcome / seats, first_distribution))
        sntv_propor.append(utils.kl_divergence(sntv_outcome / seats, first_distribution))

    return np.array(stv_propor).mean(), np.array(sntv_propor).mean()


if __name__ == '__main__':
    main()
    # preferences = PreferenceCreator(10000, 7, np.random.default_rng().permutation(7)).create_preferences()
    # print(preferences)
    # a = poll(preferences, 1) * 100
    # print(a)
    # b = poll(preferences, 0.0097) * 100
    # print(b)
    #
    # print(tabulate([a, b]))

