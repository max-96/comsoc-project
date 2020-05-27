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

def manipulate(preferences: np.ndarray, dead_parties, beta):
    ballots = np.copy(preferences)
    n_voters = preferences.shape[0]
    n_sample = int(round(n_voters * beta))
    indices = np.random.default_rng().permutation(n_voters)[:n_sample]

    for manip in indices:
        while ballots[manip, 0] in dead_parties:
            ballots[manip] = np.roll(ballots[manip], shift=-1)
    
    return ballots


def get_first_choice_dist(preferences: np.ndarray):
    return np.bincount(preferences[:, 0], minlength=preferences.shape[1]) / preferences.shape[0]

def test_beta(electoral_threshold, m, n, n_tests, political_spectrum, seats, poll_covid):
    props = []
    # progressbar = tqdm(total=50)
    betass = np.linspace(0.01, 1, num=10)
    for i, beta in enumerate(betass):
        print(i)
        # progressbar.update()

        a = lalala(electoral_threshold, m, n, n_tests, political_spectrum, seats, beta, poll_covid)
        props.append(a)
    # progressbar.close()
    mean_stv_prop, mean_sntv_prop = list(zip(*props))
    a = np.array(mean_stv_prop)
    b = np.array(mean_sntv_prop)
    print(a, b)
    plt.plot(betass, a, label='stv')
    plt.plot(betass, b, label='sntv')
    plt.legend()
    plt.show()


def test_alpha(electoral_threshold, m, n, n_tests, political_spectrum, seats, poll_covid):
    props = []
    # progressbar = tqdm(total=50)
    alphass = np.linspace(0.01, 1, num=10)
    for i, alpha in enumerate(alphass):
        print(i)
        # progressbar.update()
        astv_propor = []
        astv_l_propor = []
        for test_nr in range(n_tests):
            true_preferences = PreferenceCreator(n, m, political_spectrum).create_preferences()
            poll_results = get_poll(true_preferences, poll_covid)
            man_ballot = manipulate(true_preferences, np.nonzero(poll_results<electoral_threshold)[0], 1.)
            first_distribution = get_first_choice_dist(true_preferences)
        
            # a-STV outcome
            astv_scores, *_ = voting.alpha_STV_scores(true_preferences, alpha, electoral_threshold, percentage=True)
            astv_outcome = apportion.largest_remainder(astv_scores, seats)

            # a-STV liars outcome
            astv_l_scores, *_ = voting.alpha_STV_scores(man_ballot, alpha, electoral_threshold, percentage=True)
            astv_l_outcome = apportion.largest_remainder(astv_l_scores, seats)

            astv_propor.append(utils.kl_divergence(astv_outcome / seats, first_distribution))
            astv_l_propor.append(utils.kl_divergence(astv_l_outcome / seats, first_distribution))

        a = np.array(astv_propor).mean()
        al = np.array(astv_l_propor).mean()

        # a = lalala(electoral_threshold, m, n, n_tests, political_spectrum, seats, beta, 0.01)
        props.append((a, al))
    # progressbar.close()
    mean_astv_prop, mean_astv_l_prop = list(zip(*props))
    a = np.array(mean_astv_prop)
    b = np.array(mean_astv_l_prop)
    # print(a, b)
    plt.plot(alphass, a, label='a-stv')
    plt.plot(alphass, b, label='a-stv-liars')
    plt.legend()
    plt.show()
    


def main():
    n_tests = 20
    m = 20
    n = 1000
    seats = 100
    electoral_threshold = 0.05
    poll_covid = 0.01
    # political_spectrum = np.array([5, 4, 6, 3, 7, 2, 8, 1, 9, 0])
    political_spectrum = np.array([10,9,11,8,12,7,13,6,14,5,15, 4, 16, 3, 17, 2, 18, 1, 19, 0])
    # political_spectrum = np.arange(m)

    # test_beta(electoral_threshold, m, n, n_tests, political_spectrum, seats, poll_covid)
    test_alpha(electoral_threshold, m, n, n_tests, political_spectrum, seats, poll_covid)

    
def lalala(electoral_threshold, m, n, n_tests, political_spectrum, seats, beta, poll_covid):
    stv_propor = []
    sntv_propor = []
    for test_nr in range(n_tests):
        true_preferences = PreferenceCreator(n, m, political_spectrum).create_preferences()
        poll_results = get_poll(true_preferences, poll_covid)
        man_ballot = manipulate(true_preferences, np.nonzero(poll_results<electoral_threshold)[0], beta)
      
        # STV outcome
        # TODO reestablish the natural order
        stv_scores, *_ = voting.STV_scores(true_preferences, electoral_threshold, percentage=True)
        stv_outcome = apportion.largest_remainder(stv_scores, seats)

        # SNTV outcome
  
        sntv_scores, *_ = voting.SNTV_scores(man_ballot, electoral_threshold, percentage=True)
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

