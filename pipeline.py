from voting import *
from apportion import *
from utils import PreferenceCreator
from tabulate import tabulate

n = 30
m = 8
e = 0.1
seats = 3


def poll(preferences: np.ndarray, probability: float):
    assert 0. <= probability <= 1., "{:.4f} is not a probability".format(probability)
    n_voters = preferences.shape[0]
    n_sample = round(n_voters * probability)
    indices = np.random.default_rng().permutation(n_voters)[:n_sample]

    samples_prefs = preferences[indices, 0]
    results = np.bincount(samples_prefs, minlength=preferences.shape[1]) / n_sample
    return results


if __name__ == '__main__':
    preferences = PreferenceCreator(10000, 7, np.random.default_rng().permutation(7)).create_preferences()
    print(preferences)
    a = poll(preferences, 1)*100
    print(a)
    b = poll(preferences, 0.0097)*100
    print(b)

    print(tabulate([a,b]))
