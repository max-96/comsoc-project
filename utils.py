import numpy as np
from scipy.stats import betabinom


class PreferenceCreator:
    def __init__(self, n, m, political_spectrum):
        self.m = m
        self.n = n
        self.political_spectrum = political_spectrum

    def create_preferences(self):
        # political_spectrum = np.random.default_rng().permutation(m)
        first_choices = self.political_spectrum[betabinom.rvs(self.m - 1, 0.7, 2, size=self.n)]
        k = np.bincount(first_choices, minlength=self.m)
        preferences = []
        for i, v in enumerate(k):
            preferences.extend(self.complete_preference(i, v))

        return np.array(preferences)

    def complete_preference(self, first_choice, n_voters, constant=0):
        m = self.m
        random = np.random.default_rng()
        distances = np.abs(np.arange(m) - first_choice)  # example [0, 1, 2...m-1]
        distances = np.delete(distances, first_choice)  # example [0, 1, 2...m-1]
        pdf_unnorm = np.exp(- distances) + constant

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
