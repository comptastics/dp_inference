from scipy.linalg import sqrtm
import copy
import argparse
from dp_inference.common_dist_framework import *
from dp_inference.estimators import *
from dp_inference.priv_mech import *
import numpy as np
import math
import scipy.stats as ss

"""
The function has been modified from the original so that it runs bag of little bootstrap
only for 1D estimates. The estimator argument passes the estimator function to be used,
we need to ensure that it outputs a scalar.
"""


def general_valid_dp(
    data,
    s,
    runs_per_boot,
    cov_cov_u,
    cov_c,
    cov_r,
    mean_c,
    mean_r,
    t_cov,
    t_mean,
    rho,
    rho_mean_budget_prop,
    rho_cov_budget_prop,
    beta,
    beta_mean_budget_prop,
    beta_cov_budget_prop,
    ci_alphas,
    estimator,
    **kwargs,
):
    n = len(data)

    # Check that the budget proportions are valid
    rho_budget_sum = np.sum([rho_mean_budget_prop, rho_cov_budget_prop])
    beta_budget_sum = np.sum([beta_mean_budget_prop, beta_cov_budget_prop])
    if np.abs(rho_budget_sum - 1) > 10**-5:
        raise Exception(
            "mean and cov proportions for rho budget sum to {0} -- they must sum to 1".format(
                rho_budget_sum
            )
        )
    if np.abs(beta_budget_sum - 1) > 10**-5:
        raise Exception(
            "mean and cov proportions for beta budget sum to {0} -- they must sum to 1".format(
                beta_budget_sum
            )
        )

    # allocate privacy budget
    rho_mean = rho * rho_mean_budget_prop
    rho_cov = rho * rho_cov_budget_prop

    # allocate failure budget
    beta_mean = beta * beta_mean_budget_prop
    beta_cov = beta * beta_cov_budget_prop

    blb_mean_ests, blb_cov_ests = blb_means_covs_1D(
        data=data, b=int(n / s), runs_per_boot=runs_per_boot, estimator=estimator
    )
    cov_uppers = blb_cov_ests
    beta_cov = beta_cov / 2
    upper_bound_cov_beta = beta_cov

    beta_cov_est = (
        beta_cov / 2
    )  # half for estimation, half for Loewner upper bounding after the fact

    _, privatization_variances, _, priv_cov_ests = multivariate_mean_iterative(
        cov_uppers,
        c=cov_c,
        r=cov_r,
        cov=cov_cov_u,
        t=t_cov,
        betas=[1 / (4 * (t_cov - 1)) * beta_cov_est] * (t_cov - 1)
        + [(3 / 4) * beta_cov_est],
        Rhos=[1 / (4 * (t_cov - 1)) * rho_cov] * (t_cov - 1) + [(3 / 4) * rho_cov],
        s=s,
    )

    est_priv_cov, est_priv_cov_covariance = inverse_covariance_weighting(
        priv_cov_ests, [v for v in privatization_variances]
    )
    """
    convert estimated covariance cells back to actual matrix
    """
    if est_priv_cov < 0:
        est_priv_cov = 0

    privatization_var = est_priv_cov_covariance
    sd = np.sqrt(privatization_var)

    # NOTE: version below is for diagonal cov
    est_priv_cov += ss.norm.ppf(
        (1 - upper_bound_cov_beta / 1),
        loc=0,
        scale=np.sqrt(privatization_var),
    )
    # step instead of PSD projection
    est_priv_cov = max(0, est_priv_cov)

    """ Mean estimation """

    _, privatization_variances, _, sa_means = multivariate_mean_iterative(
        blb_mean_ests,
        c=mean_c,
        r=mean_r,
        cov=s * est_priv_cov,
        t=t_mean,
        betas=[1 / (4 * (t_mean - 1)) * beta_mean] * (t_mean - 1)
        + [(3 / 4) * beta_mean],
        Rhos=[1 / (4 * (t_mean - 1)) * rho_mean] * (t_mean - 1) + [(3 / 4) * rho_mean],
        s=s,
    )

    covariances_of_mean_est = [
        priv_var + est_priv_cov for priv_var in privatization_variances
    ]
    est_priv_mean, est_priv_mean_covariance = inverse_covariance_weighting(
        sa_means, covariances_of_mean_est
    )
    priv_var = est_priv_mean_covariance

    quantile = ss.norm.ppf(
        1 - ci_alphas / 2, loc=0, scale=np.sqrt(est_priv_cov + priv_var)
    )
    ci_ls = est_priv_mean - quantile
    ci_us = est_priv_mean + quantile

    cis = [(ci_ls, ci_us)]

    return est_priv_mean, priv_var, est_priv_cov, cis


# Utils type functions


def parse_args():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    return opt


def eps_to_rho(epsilon, delta):
    return (np.sqrt(epsilon + np.log(1 / delta)) - np.sqrt(np.log(1 / delta))) ** 2


def gaussian_tailbound(d, b):
    """
    Calculate 1 - b probability upper bound on the L2 norm of a draw
    from a d-dimensional standard multivariate Gaussian

    Args:
        d (positive int): dimension of the multivariate Gaussian in question
        b (float in (0,1)): probability that draw from the distribution has a
                            larger L2 norm than our bound

    Returns:
        1-b probability upper bound on the L2 norm
    """
    return (d + 2 * (d * math.log(1 / b)) ** 0.5 + 2 * math.log(1 / b)) ** 0.5


def inverse_covariance_weighting(means, covariances):
    """
    Takes a set of estimates of a single quantity and weights them by the inverse of their (co)variances
    to get the minimum (co)variance combined estimate.

    General idea explained here (https://en.wikipedia.org/wiki/Inverse-variance_weighting),
    statement for the multivariate case found in Theorem 4.1 here (https://arxiv.org/pdf/2110.14465.pdf)

    Args:
        means (vector of length k): d-dimensional parameter estimates
        covariances (vector of length k): d x d covariance estimates

    Returns:
        optimal estimate (est) and associated covariance of the new estimate (inverse_sum)
    """

    """first term"""
    # get inverse of each covariance matrix
    inverse_covs = [1 / cov_i for cov_i in covariances]

    # add inverses together
    sum_inverse_covs = np.sum(inverse_covs)

    # get inverse of sum
    inverse_sum = 1 / sum_inverse_covs

    """second term"""
    # get element-wise product of inverse covariances and means and sum them
    cov_mean_prod = [
        np.dot(inverse_cov, mean) for inverse_cov, mean in zip(inverse_covs, means)
    ]
    cov_mean_prod_sum = np.sum(cov_mean_prod)

    """return est and covariance"""
    est = np.dot(inverse_sum, cov_mean_prod_sum)
    return (est, inverse_sum)


# BLB functions
def blb_means_covs_1D(data, b, runs_per_boot, estimator):
    """
    Runs bag of little bootstraps particularly for 1D mean estimation.

    Args:
        subsampled_data (pandas dataframe): Data on which blb should be run.
        overall_n (int): Number of elements in overall data (note that this is not the number of elements
                         in the subsampled data).
        r (int): Number of simulations to perform.
        estimator (function): Function we are using the blb to estimate. The estimator should take
                              the subsampled data as the first argument.
        scalable (bool): Whether or not estimator is scalable with weights argument.
        **kwargs: Arguments to estimator function.

    Returns:
        float or list of floats: Averaged results of estimator function applied
    """
    n = data.shape[0]

    s = int(n / b)
    perm = np.random.permutation(n)
    # returns a permutation of X
    data_perm = data[perm]
    sample_idx = np.arange(s)

    mean_arr = []
    cov_arr = []

    for i in sample_idx:
        data_b = data_perm[i * b : (i + 1) * b]

        est_arr = []

        for j in range(runs_per_boot):
            resampled_idxs = np.random.choice(b, n)
            resampled_data = data_b[resampled_idxs]
            est_arr.append(estimator(resampled_data))

        est_arr = np.array(est_arr)
        est_mean = np.mean(est_arr, axis=0)
        est_cov = np.cov(est_arr.T)
        mean_arr.append(est_mean)
        cov_arr.append(est_cov)

    return (mean_arr, cov_arr)


# coinpress functions
def multivariate_mean_step(X, c, r, beta, rho, s):
    """
    Privately estimate d-dimensional mean of X

    Args:
        X (pytorch tensor): data for which we want the mean
        c (float): center of ball (with radius r) which contains the mean
        r (float): radius of ball (with center c) which contains the mean
        beta (beta): failure probability (i.e. probability of clipping at least one point)
        rho (beta): privacy budget
        args (set of arguments): arguments to be passed to multivariate_mean_step -- in this version, it's just the
                                 dimensions of X (args.n and args.d)

    Returns:
        - mean estimate (new center of shrinking ball)
        - bound estimate (new radius of shrinking ball)
        - variance of privacy mechanism
        - total number of points clipped (sensitive, used only for diagnostics)
    """
    d = 1

    """
    functional beta is beta/2 because of use in gamma and r
    """
    beta = beta / 2

    """establish clipping bounds"""
    gamma_1 = gaussian_tailbound(d, beta / s)
    gamma_2 = gaussian_tailbound(d, beta)
    clip_thresh = r + gamma_1

    """clip points 1D"""
    x = X - c
    mag_x = np.abs(x)
    outside_ball = mag_x > clip_thresh
    total_clipped = np.sum(outside_ball)
    x_hat = np.sign(x)  # Only works for 1D
    if np.sum(outside_ball) > 0:
        X[outside_ball] = c + (x_hat[outside_ball] * clip_thresh)

    """calculate sensitivity"""
    delta = 2 * clip_thresh / float(s)
    sd = delta / (2 * rho) ** 0.5

    """add noise (calculate private mean) and update radius of ball"""
    Y = np.random.normal(0, sd, size=d)
    c = np.sum(X, axis=0) / float(s) + Y
    r = gamma_2 * np.sqrt(1 / float(s) + (2 * clip_thresh**2) / (s**2 * rho))

    return c, r, sd**2, total_clipped


def multivariate_mean_iterative(X, c, r, cov, t, betas, Rhos, s):
    """
    Privately estimate d-dimensional mean of X, iteratively improving the estimate over t steps

    Args:
        X (pytorch tensor): data for which we want the mean
        c (float): center of ball (with radius r) which contains the mean
        r (float): radius of ball (with center c) which contains the mean
        cov (matrix): Loewner upper bound on the covariance of X
        t (int): number of iterations to run of CoinPress algorithm
        betas (list of length t): failure probability (i.e. probability of clipping at least one point) at each iteration
        Ps (list of length t): privacy budget at each iteration
        args (set of arguments): arguments to be passed to multivariate_mean_step -- in this version, it's just the
                                 dimensions of X (args.n and args.d)

    Returns:
        - final release of CoinPress (releasable but not used, this was the algorithm output in the original CoinPress paper but we now release
                                      all t iterations and combine them via inverse covariance weighting)
        - variance of privacy mechanism at each iteration (releasable, is used)
        - total number of points clipped at each iteration (sensitive, but useful for testing)
        - CoinPress release from each of the t iterations (releasable, is used)
    """

    """standardize data to have empirical mean 0 and covariance upper bounded by the identity matrix (assuming cov argument is properly set)"""
    means = np.mean(np.array(X), axis=0)
    sds = np.sqrt(cov)
    sqrt_cov = sds  # For 1D, this is just the standard deviation
    inv_sqrt_cov = 1 / sqrt_cov

    X = (X - means) * inv_sqrt_cov

    c = c / sds
    r = r / sds

    """run CoinPress for t iterations"""
    cs = []
    rs = []
    noise_variances = []
    total_clipped_vec = []
    for i in range(t - 1):
        c, r, noise_variance, total_clipped = multivariate_mean_step(
            X.copy(), c, r, betas[i], Rhos[i], s
        )
        cs.append(c)
        rs.append(r)
        noise_variances.append(noise_variance)
        total_clipped_vec.append(total_clipped)

    c, r, last_step_noise_variance, total_clipped = multivariate_mean_step(
        X.copy(), c, r, betas[t - 1], Rhos[t - 1], s
    )
    cs.append(c)
    rs.append(r)
    noise_variances.append(last_step_noise_variance)
    total_clipped_vec.append(total_clipped)

    """scale estimates back from mean 0 and covariance <= identity"""

    cs = [c * sqrt_cov + means for c in cs]
    priv_vars = [(np.sqrt(noise_var) * sqrt_cov) ** 2 for noise_var in noise_variances]

    return (cs[-1], priv_vars, total_clipped_vec, cs)
