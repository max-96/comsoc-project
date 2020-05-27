from typing import Tuple

import numpy as np
from scipy.stats import betabinom


def kl_divergence(parliament: np.ndarray, true_pref: np.ndarray, eps: float = 1e-16) -> float:
    assert parliament.shape == true_pref.shape
    return np.sum(np.where(parliament != 0, parliament * np.log2(parliament / true_pref), 0))

def governability(parliament: np.ndarray) -> Tuple[int, int]:
    m = parliament.shape[0]
    v = np.zeros_like(parliament)

    for alt in range(m):
        v[alt] = parliament[alt]
        if v[alt] > 0.5:
            return alt, 1

    for step in range(2, m):
        for alt in range(m - step + 1):
            v[alt] = parliament[alt] + v[alt + 1]
            if v[alt] > 0.5:
                return alt, step
    return -1, -1


class PreferenceCreator:
    def __init__(self, n, m, political_spectrum, alpha=0.2, beta=0.5, c=0):
        self.m = m
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.political_spectrum = political_spectrum

    def create_preferences(self):
        # political_spectrum = np.random.default_rng().permutation(m)
        first_choices = self.political_spectrum[betabinom.rvs(self.m - 1, self.alpha, self.beta, size=self.n)]
        k = np.bincount(first_choices, minlength=self.m)[:self.m]
        preferences = []
        for i, v in enumerate(k):
            preferences.extend(self.complete_preference(i, v))

        return np.array(preferences)

    def complete_preference(self, first_choice, n_voters):
        m = self.m
        assert first_choice < m, f'Broken first choice: fc: {first_choice} m: {m}'
        random = np.random.default_rng()
        distances = np.abs(np.arange(m) - first_choice)  # example [0, 1, 2...m-1]
        distances = np.delete(distances, first_choice)  # example [0, 1, 2...m-1]
        pdf_unnorm = np.exp(- distances) + self.c

        completion = []
        for i in range(n_voters):
            complement_voter = [first_choice]
            indices = np.delete(np.arange(m), first_choice)  # contains the party of each position in the pdf
            pdf = np.copy(pdf_unnorm)
            for x in range(m - 1):
                pdf /= pdf.sum()
                cdf = np.cumsum(pdf)
                selected_index = np.searchsorted(cdf, random.random())
                selected_party = indices[selected_index]
                complement_voter.append(selected_party)
                indices = np.delete(indices, selected_index)
                pdf = np.delete(pdf, selected_index)
            completion.append(complement_voter)
        return completion


if __name__ == '__main__':
    political_spectrum = np.random.default_rng().permutation(10)
    prefmaker = PreferenceCreator(20, 10, political_spectrum)
    print(np.array(prefmaker.create_preferences()))
    # print(len(prefmaker.complete_preference(5, 1)[0]))
