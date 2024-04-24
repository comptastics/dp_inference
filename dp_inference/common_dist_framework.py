import numpy as np
from scipy.stats import weibull_min as weibull
from scipy.stats import pareto
from scipy.stats import chi2
from scipy.stats import skewnorm
from scipy.stats import truncnorm

from scipy.stats import triang
import copy
from abc import ABC, abstractmethod

### DISTRIBUTIONS ###


class Distribution(ABC):
    @abstractmethod
    def sample(n):
        pass


class Gauss1D(Distribution):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def sample(self, n):
        return np.random.normal(self.mu, self.sigma, size=(n, 1))

    def get_mean(self):
        return self.mu

    def get_median(self):
        return self.mu

    def get_d(self):
        return 1

    def get_std(self):
        return self.sigma

    def get_bounds(self):
        return -np.inf, np.inf


class GaussmultiD(Distribution):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, size=n)

    def get_mean(self):
        return self.mu

    def get_median(self):
        return self.mu

    def get_d(self):
        return len(self.mu)


class TruncGauss1D(Distribution):
    def __init__(self, trunc_left, trunc_right, mu=0, sigma=1):
        self.a, self.b, self.mu, self.sigma = trunc_left, trunc_right, mu, sigma

    # 1 D output enforced
    def sample(self, n):
        return truncnorm.rvs(self.a, self.b, loc=self.mu, scale=self.sigma, size=(n, 1))

    def get_mean(self):
        return truncnorm.stats(self.a, self.b, loc=self.mu, scale=self.sigma)[0]

    def get_std(self):
        return np.sqrt(
            truncnorm.stats(self.a, self.b, loc=self.mu, scale=self.sigma)[1]
        )

    def get_median(self):
        return truncnorm.median(self.a, self.b, loc=self.mu, scale=self.sigma)

    def get_d(self):
        return 1

    def get_med_std(self):
        return 1 / (
            2
            * truncnorm.pdf(
                self.get_median(), self.a, self.b, loc=self.mu, scale=self.sigma
            )
        )

    def get_bounds(self):
        return self.a * self.sigma, self.b * self.sigma


class Triang1D(Distribution):
    def __init__(self, c, left=0, right=1):
        self.c, self.left, self.right = c, left, right

    # 1 D output enforced
    def sample(self, n):
        return triang.rvs(
            c=self.c, loc=self.left, scale=self.right - self.left, size=(n, 1)
        )

    def get_mean(self):
        return triang.stats(c=self.c, loc=self.left, scale=self.right - self.left)[0]

    def get_std(self):
        return np.sqrt(
            triang.stats(c=self.c, loc=self.left, scale=self.right - self.left)[1]
        )

    def get_median(self):
        return triang.median(c=self.c, loc=self.left, scale=self.right - self.left)

    def get_d(self):
        return 1

    def get_med_std(self):
        return 1 / (
            2
            * triang.pdf(
                self.get_median(), c=self.c, loc=self.left, scale=self.right - self.left
            )
        )

    def get_bounds(self):
        return self.left, self.right


class MultidimMixOfTwoGauss(Distribution):
    def __init__(self, d, mu, sigma, p):
        self.d, self.mu, self.sigma, self.p = d, mu, sigma, p

    def sample(self, n):
        samples = []
        for i in range(n):
            p_hat = np.random.uniform()
            if p_hat <= self.p:
                x = np.random.normal(self.mu, self.sigma, size=self.d)
            else:
                x = np.random.normal(0, 1, size=self.d)

            samples.append(x)

        return np.array(samples)

    def get_mean(self):
        return self.p * self.mu

    def get_d(self):
        return self.d


class MixOfTwoGauss1D(Distribution):
    def __init__(self, mu, sigma, p):
        self.mu, self.sigma, self.p = mu, sigma, p

    def sample(self, n):
        samples = []
        for i in range(n):
            p_hat = np.random.uniform()
            if p_hat <= self.p:
                x = np.random.normal(self.mu, self.sigma)
            else:
                x = np.random.normal()

            samples.append([x])

        return np.array(samples)

    def get_mean(self):
        return self.p * self.mu

    def get_d(self):
        return 1


class Weibull(Distribution):
    def __init__(self, c):
        self.c = c

    def sample(self, n):
        return weibull.rvs(self.c, size=(n, 1))

    def get_mean(self):
        return weibull.mean(self.c)

    def get_median(self):
        return weibull.median(self.c)

    def get_d(self):
        return 1


class Pareto(Distribution):
    def __init__(self, b):
        self.b = b

    def sample(self, n):
        return pareto.rvs(self.b, size=(n, 1))

    def get_mean(self):
        return pareto.mean(self.b)

    def get_median(self):
        return pareto.median(self.b)

    def get_d(self):
        return 1


class Chi2(Distribution):
    def __init__(self, df):
        self.df = df

    def sample(self, n):
        return chi2.rvs(self.df, size=(n, 1))

    def get_mean(self):
        return chi2.mean(self.df)

    def get_median(self):
        return chi2.mean(self.df)

    def get_d(self):
        return 1


class Skewnorm(Distribution):
    def __init__(self, a):
        self.a = a

    def sample(self, n):
        return skewnorm.rvs(self.a, size=(n, 1))

    def get_mean(self):
        return skewnorm.mean(self.a)

    def get_median(self):
        return skewnorm.median(self.a)

    def get_d(self):
        return 1


### CONFIDENCE SETS ###


class ConfidenceSet(ABC):
    @abstractmethod
    def contains(x):
        pass


class Box(ConfidenceSet):
    def __init__(self, l=None, u=None):
        self.l = l if l is not None else -np.infty
        self.u = u if u is not None else np.infty

    def contains(self, x):
        if np.array(x <= self.u).all() and np.array(x >= self.l).all():
            return True
        else:
            return False


class Box1D(Box):
    def __init__(self, l=None, u=None):
        super().__init__(l, u)

    def __lt__(self, x):
        return self.u < x

    def __gt__(self, x):
        return self.l > x

    def contains(self, x):
        if x <= self.u and x >= self.l:
            return True
        else:
            return False


class Threshold(Box):
    def __init__(self, t):
        super().__init__(u=t)


### HISTOGRAMS ###


class Histogram(ABC):
    def __init__(self, sets):
        # Assumes sets are indexed by two indices with each element being a ConfidenceSet object
        # For finding confidence sets, sets formed using hadamard construction
        # For coverage probability, can just input 1 set
        # can parallelly pass sets (say 4), will take 4x time to add.

        self.sets = sets
        self.histogram = self.init_histogram()

        self.total = 0

    def init_histogram(self):
        # complexity = first dimension of sets (last level of tree)
        histogram = []
        for i in range(len(self.sets)):
            histogram.append(np.zeros(len(self.sets[i])))

        return histogram

    def get_histogram(self, normalized=True):
        # complexity = first dimension of sets
        if normalized:
            normalized_histogram = copy.deepcopy(self.histogram)
            for i in range(len(self.sets)):
                normalized_histogram[i] /= self.total
            return normalized_histogram
        else:
            return copy.deepcopy(self.histogram)

    def get_sets(self):
        return copy.deepcopy(self.sets)

    def add(self, x):
        # complexity  =  number of sets
        self.total += 1

        for i in range(len(self.sets)):
            for j in range(len(self.sets[i])):
                if self.sets[i][j].contains(x):
                    self.histogram[i][j] += 1


class Hadamard(Histogram):
    @abstractmethod
    def predict(self, A):
        pass


class Hadamard1D(Hadamard):
    # structure of sets: first dimension is depths with big set first
    # second dimension is bins from left to right, last row has most bins

    def __init__(self, min, max, precision, round_up=False, round_dec=5):
        self.min, self.max = min, max
        self.range = self.max - self.min
        self.precision = precision
        self.round_up = round_up
        self.round_dec = round_dec

        self.sets = self.init_sets()
        self.histogram = self.init_histogram()

        self.total = 0

    def init_sets(self):
        n_leaf_bins = int(self.range / self.precision)
        self.depth = int(np.ceil(np.log2(n_leaf_bins)))

        if self.round_up == True:
            n_leaf_bins = 2**self.depth
            self.precision = self.range / n_leaf_bins

        sets = [[] for i in range(self.depth)]
        for i in range(self.depth):
            bin_size = pow(2, i) * self.precision
            n_bins = int(np.ceil(self.range / bin_size))

            for j in range(n_bins):
                l = bin_size * j + self.min
                u = bin_size * (j + 1) + self.min

                sets[self.depth - i - 1].append(Box1D(l, u))

        return sets

    def add(self, x):
        # complexity = depth (~log range)
        self.total += 1
        if x >= self.min and x < self.max:
            bstr = "{0:b}".format(int((x - self.min) / self.precision)).zfill(
                self.depth
            )

            leaf_bin = 0
            for i in range(self.depth):
                leaf_bin *= 2
                if bstr[i] == "1":
                    leaf_bin += 1

            bin = leaf_bin
            for i in range(self.depth):
                # assert self.sets[self.depth-i-1][bin].contains(x)
                self.histogram[self.depth - i - 1][bin] += 1

                bin = int(bin / 2)

    def predict(self, A):
        active = [[0 for j in range(len(self.sets[i]))] for i in range(self.depth)]

        # last layer of binary tree
        for j in range(len(self.sets[self.depth - 1])):
            if (
                self.sets[self.depth - 1][j].l >= A.l
                and self.sets[self.depth - 1][j].u <= A.u
            ):
                active[self.depth - 1][j] = 1

        for i in range(1, self.depth):
            for j in range(len(self.sets[self.depth - i - 1])):
                if active[self.depth - i][2 * j] == 1 and (
                    2 * j + 1 == len(self.sets[self.depth - i])
                    or active[self.depth - i][2 * j + 1] == 1
                ):
                    active[self.depth - i][2 * j] = 0
                    if 2 * j + 1 < len(self.sets[self.depth - i]):
                        active[self.depth - i][2 * j + 1] = 0

                    active[self.depth - i - 1][j] = 1

        histogram = self.get_histogram()
        cov_prob = 0

        for i in range(self.depth):
            for j in range(len(self.sets[i])):
                if active[i][j] == 1:
                    cov_prob += histogram[i][j]

        return cov_prob


class IntervalDiv(Histogram):
    def __init__(self, min, max, precision, round_up=False, round_dec=13):
        self.min, self.max = min, max
        self.range = self.max - self.min
        self.precision = precision
        self.round_up = round_up
        self.round_dec = round_dec

        (
            self.l_arr,
            self.u_arr,
        ) = self.init_sets()
        self.histogram = self.init_histogram()

        self.total = 0

    def init_sets(self):
        if self.round_up == True:
            n_leaf_bins = int(self.range / self.precision)
            self.depth = int(np.ceil(np.log2(n_leaf_bins)))

            n_leaf_bins = 2**self.depth
            self.precision = self.range / n_leaf_bins

        big_arr = np.around(
            np.arange(self.min, self.max + self.precision, self.precision),
            self.round_dec,
        )
        l_arr = big_arr[:-1]
        u_arr = big_arr[1:]

        return l_arr, u_arr

    def init_histogram(self):
        return np.zeros(len(self.l_arr))

    def add(self, x):
        # complexity = depth (~log range)
        self.total += 1
        if x >= self.min and x < self.max:
            idx = np.searchsorted(self.u_arr, x, side="right")
            self.histogram[idx] += 1

    def add_batch(self, X):
        # complexity = depth (~log range)
        self.total += len(X)
        if not type(X) == np.ndarray:
            X = np.array(X)
        X_clean = X[(X >= self.min) & (X <= self.max)]
        idx_lst = np.searchsorted(self.u_arr, X_clean, side="right")
        self.histogram += np.bincount(idx_lst, minlength=self.histogram.size)

    def get_histogram(self, normalized=True):
        # complexity = first dimension of sets
        if normalized:
            normalized_histogram = copy.deepcopy(self.histogram)
            normalized_histogram /= self.total
            return normalized_histogram
        else:
            return copy.deepcopy(self.histogram)

    def predict(self, A):
        l_ind = np.searchsorted(self.l_arr, A.l, side="left")
        u_ind = np.searchsorted(self.u_arr, A.u, side="right")

        histogram = self.get_histogram()

        cov_prob = np.sum(histogram[l_ind:u_ind])
        return cov_prob


def cov_predict_int_batch(A, hist_structure, hist_lst):
    """Output the predicted coverage probability
    of set A using the estimated histogram

    Args:
        A (Set): The set of which you want the coverage probability estimate
                 Usually a 1D Box
        hist_structure (IntervalDiv): structure of the histogram
        hist_lst (list): list of 1D array containing pdf values in histogram buckets

    Returns:
        float: estimated coverage probability
    """
    l_ind = np.searchsorted(hist_structure.l_arr, A.l, side="left")
    u_ind = np.searchsorted(hist_structure.u_arr, A.u, side="right")

    hist_lst = np.array(hist_lst)
    cov_prob = np.sum(hist_lst[:, l_ind:u_ind], axis=1)

    return cov_prob


def cov_predict(A, hist_structure, histogram):
    """Output the predicted coverage probability
    of set A using the estimated histogram

    Args:
        A (Set): The set of which you want the coverage probability estimate
                 Usually a 1D Box
        sets (_type_): sets describing the histogram
        depth (_type_): Depth of the histogram
        histogram (_type_): 2D array containing pdf values in histogram buckets

    Returns:
        float: estimated coverage probability
    """
    sets = hist_structure.sets
    depth = hist_structure.depth

    active = [[0 for j in range(len(sets[i]))] for i in range(depth)]

    # last layer of binary tree
    for j in range(len(sets[depth - 1])):
        if sets[depth - 1][j].l >= A.l and sets[depth - 1][j].u <= A.u:
            active[depth - 1][j] = 1

    for i in range(1, depth):
        for j in range(len(sets[depth - i - 1])):
            if active[depth - i][2 * j] == 1 and (
                2 * j + 1 == len(sets[depth - i]) or active[depth - i][2 * j + 1] == 1
            ):
                active[depth - i][2 * j] = 0
                if 2 * j + 1 < len(sets[depth - i]):
                    active[depth - i][2 * j + 1] = 0

                active[depth - i - 1][j] = 1

    cov_prob = 0

    for i in range(depth):
        for j in range(len(sets[i])):
            if active[i][j] == 1:
                cov_prob += histogram[i][j]

    return cov_prob


def cov_predict_batch(A, hist_structure, hist_list):
    """Output the predicted coverage probability
    of set A using the estimated histogram

    Args:
        A (Set): The set of which you want the coverage probability estimate
                 Usually a 1D Box
        sets (_type_): sets describing the histogram
        depth (_type_): Depth of the histogram
        hist_list (_type_): List of 2D arrays containing pdf values in histogram buckets

    Returns:
        float: estimated coverage probability
    """
    sets = hist_structure.sets
    depth = hist_structure.depth

    n_col = len(sets[depth - 1])
    active = np.zeros((depth, n_col))

    # last layer of binary tree
    for j in range(n_col):
        if sets[depth - 1][j].l >= A.l and sets[depth - 1][j].u <= A.u:
            active[depth - 1, j] = 1

    for i in range(1, depth):
        for j in range(len(sets[depth - i - 1])):
            if active[depth - i, 2 * j] == 1 and (
                2 * j + 1 == len(sets[depth - i]) or active[depth - i, 2 * j + 1] == 1
            ):
                active[depth - i, 2 * j] = 0

                if 2 * j + 1 < len(sets[depth - i]):
                    active[depth - i, 2 * j + 1] = 0

                active[depth - i - 1, j] = 1

    cov_prob_list = [0 for _ in range(len(hist_list))]
    # get list of indices with nonzero values in active
    nonzero = np.nonzero(active)
    for i in range(len(nonzero[0])):
        for hist_num, histogram in enumerate(hist_list):
            cov_prob_list[hist_num] += histogram[nonzero[0][i]][nonzero[1][i]]

    return cov_prob_list


def conf_set_quick(alpha, hist_structure, histogram):
    """Calculates the confidence set by finding the
     1 - alpha/2 and alpha/2 quantiles by searching over the last level of the histogram

    Args:
        alpha (float): (1 - alpha) is the
                     coverage of confidence set we want
        min (float): Lower end of search range
        max (float): Upper end of search range
        sets (sets): sets describing the histogram
        depth (int): Depth of the histogram
        histogram (2D List): 2D array containing pdf values in histogram buckets

    Returns:
        Box1D: A confidence set estimate with coverage 1 - alpha
    """
    sets = hist_structure.sets
    depth = hist_structure.depth
    min = sets[-1][0].l
    max = sets[-1][-1].u
    lower = min
    upper = max
    for i in range(len(sets[depth - 1])):
        upp = sets[depth - 1][i].u
        A = Box1D(l=min, u=upp)
        if cov_predict(A, hist_structure, histogram) >= alpha / 2:
            lower = sets[depth - 1][i].l
            break

    for i in range(len(sets[depth - 1]) - 1, -1, -1):
        low = sets[depth - 1][i].l
        A = Box1D(l=low, u=max)
        if cov_predict(A, hist_structure, histogram) >= alpha / 2:
            upper = sets[depth - 1][i].u
            break

    return Box1D(l=lower, u=upper)


def conf_set_short(alpha, hist_structure, histogram):
    """Compute the shortest set with a given coverage or an estimated set with the given coverage
    The estimated set is found by finding all valid sets and outputting the median set.

    Args:
       alpha (float): (1 - alpha) is the
                     coverage of confidence set we want
        min (float): Lower end of search range
        max (float): Upper end of search range
        sets (sets): sets describing the histogram
        depth (int): Depth of the histogram
        histogram (2D List): 2D array containing pdf values in histogram buckets
    Returns:
        Box1D,Box1D: The shortest set and a median length set with the given coverage
    """
    valid_sets = []
    sets = hist_structure.sets
    depth = hist_structure.depth
    min = sets[-1][0].l
    max = sets[-1][-1].u
    prev_low = min
    prev_upp = max
    prev_cov = 1
    for i in range(len(sets[depth - 1])):
        low = sets[depth - 1][i].l
        for j in range(len(sets[depth - 1]) - 1, i, -1):
            upp = sets[depth - 1][j].u
            A = Box1D(l=low, u=upp)
            covg = cov_predict(A, hist_structure, histogram)
            if covg < 1 - alpha and prev_cov >= 1 - alpha:
                new_low = prev_low + (sets[depth - 1][j].u - sets[depth - 1][j].l)
                new_covg = cov_predict(
                    Box1D(l=new_low, u=prev_upp), hist_structure, histogram
                )
                if new_covg < 1 - alpha:
                    valid_sets.append([prev_low, prev_upp, prev_upp - prev_low])
                    prev_low = low
                    prev_upp = upp
                    prev_cov = covg
                    break
            prev_low = low
            prev_upp = upp
            prev_cov = covg

    sorted_valid_sets = sorted(valid_sets, key=lambda l: l[2])

    shortest_set = Box1D(sorted_valid_sets[0][0], sorted_valid_sets[0][1])
    est_set = Box1D(
        sorted_valid_sets[len(valid_sets) // 2][0],
        sorted_valid_sets[len(valid_sets) // 2][1],
    )
    return shortest_set, est_set


def conf_set_mid_quick(alpha, hist_structure, histogram, err_correction=0):
    """Outputs a set with coverage 1 - alpha which is
    somehow a middle set amongst all sets that have this coverage
    It lists all the top and bottom endpoints where the covg switches from below alpha/2 to
    above alpha/2 and chooses the median point in both the cases.

    Args:
        alpha (float): (1 - alpha) is the
                     coverage of confidence set we want
        min (float): Lower end of search range
        max (float): Upper end of search range
        sets (sets): sets describing the histogram
        depth (int): Depth of the histogram
        histogram (2D List): 2D array containing pdf values in histogram buckets

    Returns:
        Box1D: A confidence set estimate with coverage 1 - alpha
    """
    sets = hist_structure.sets
    depth = hist_structure.depth
    min = sets[-1][0].l
    max = sets[-1][-1].u
    alpha = alpha + err_correction
    prev_cov = 0
    lower_lst = []
    for i in range(len(sets[depth - 1])):
        upp = sets[depth - 1][i].u
        A = Box1D(l=min, u=upp)
        covg = cov_predict(A, hist_structure, histogram)
        if covg >= alpha / 2 and prev_cov <= alpha / 2:
            lower_lst.append(sets[depth - 1][i].l)
        prev_cov = covg

    prev_cov = 0
    upper_lst = []
    for i in range(len(sets[depth - 1]) - 1, -1, -1):
        low = sets[depth - 1][i].l
        A = Box1D(l=low, u=max)
        covg = cov_predict(A, hist_structure, histogram)
        if covg >= alpha / 2 and prev_cov <= alpha / 2:
            upper_lst.append(sets[depth - 1][i].u)
        prev_cov = covg

    if lower_lst:
        lower_mid = lower_lst[len(lower_lst) // 2]
        lower_quick = lower_lst[0]
    else:
        lower_mid = min
        lower_quick = min

    if upper_lst:
        upper_mid = upper_lst[len(upper_lst) // 2]
        upper_quick = upper_lst[0]
    else:
        upper_mid = max
        upper_quick = max

    return Box1D(l=lower_quick, u=upper_quick), Box1D(l=lower_mid, u=upper_mid)


def conf_set_symm(
    alpha, hist_structure, histogram, oracle_min_length=0, err_correction=0
):
    sets = hist_structure.sets
    depth = hist_structure.depth

    precision = sets[depth - 1][0].u - sets[depth - 1][0].l
    num_int = np.max(int(oracle_min_length / precision), 0) // 2
    tot_len = len(sets[depth - 1])
    q, r = divmod(tot_len, 2)
    if r == 1:
        l_ind = q
        u_ind = q
        search_len = q
    else:
        l_ind = q - 1
        u_ind = q
        search_len = q - 1

    for i in range(num_int, search_len):
        l = sets[depth - 1][l_ind - i].l
        u = sets[depth - 1][u_ind + i].u
        A = Box1D(l, u)
        if cov_predict(A, hist_structure, histogram) > 1 - alpha + err_correction:
            return A

    return Box1D(sets[depth - 1][0].l, sets[depth - 1][-1].u)
