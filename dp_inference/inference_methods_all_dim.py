import copy
from dp_inference.common_dist_framework import (
    Box1D,
    cov_predict_batch,
    cov_predict_int_batch,
)
from dp_inference.estimators import mean1D, std_mean1D
from dp_inference.priv_mech import mod_inv_sens
import numpy as np

### INFERENCE METHODS ###


def bootstrap_stats_alldim(
    data,
    n_samples,
    n_runs,
    studentize=False,
    eps_sigma=None,
    eps_param=None,
    est=mean1D,
    std_est=None,
    priv_est=None,
    priv_std_est=None,
):
    """Function implementing bootstrap which outputs all the stats computed.
    This method is called as a subroutine by other functions either implementing
    the naive bootstrap or the bag of little bootstraps
    Args:
        data (array_like): input data
        n_samples (int): resample size
        n_runs (int): number of monte carlo runs
        studentize (bool, optional): bool for studentize or not. Defaults to False.
        eps_param (int, optional): privacy parameter for parameter estimator. Defaults to None.
        eps_sigma (int, optional): privacy parameter for std dev estimator. Defaults to None.
        est (function, optional): estimator for the parameter. Defaults to mean1D.
        std_est (function, optional): estimator for the standard deviation. Defaults to std_mean1D.
        priv_est (function, optional): private estimator for the parameter. Defaults to None.
        priv_std_est (function, optional): private estimator for the standard deviation. Defaults to None.
    Returns:
        list: list of statistics computed in all monte carlo runs
    """

    stats_lst = []
    n, d = data.shape

    d = d + 1

    if est == mean1D or std_est == std_mean1D:
        raise ValueError("Estimator is for 1D data")

    # estimating the population is always nonprivate
    theta_X = est(data)

    theta_s = []
    for run in range(n_runs):
        s = np.random.choice(n, n_samples)

        data_s = data[s]
        # Uses indices to form a subsample

        if eps_param is None:
            theta_s = est(data_s)
        else:
            theta_s = priv_est(data_s, eps_param)

        if not studentize:
            stat = np.sqrt(n_samples) * (theta_X - theta_s)

        else:
            if eps_sigma is None:
                sigma_hat = std_est(data_s)
            else:
                sigma_hat = priv_std_est(data_s, eps_sigma)

            if d > 1:
                sigma_hat_inv = np.linalg.inv(sigma_hat)
                stat = np.sqrt(n_samples) * sigma_hat_inv @ (theta_X - theta_s)
            else:
                sigma_hat_inv = 1 / sigma_hat
                stat = np.sqrt(n_samples) * sigma_hat_inv * (theta_X - theta_s)

        stats_lst.append(stat)

    return np.array(stats_lst)


def bootstrap_quantile_ci_alldim(
    data,
    n_samples,
    n_runs,
    studentize=False,
    eps_sigma=None,
    eps_param=None,
    est=mean1D,
    std_est=None,
    priv_est=None,
    priv_std_est=None,
    alpha=0.05,
):
    """
    Function implementing bootstrap which outputs the confidence interval
    computed using the quantile method.

    Args:
        data (array_like): input data
        n_samples (int): resample size
        n_runs (int): number of monte carlo runs
        dim (int, optional): dimension to compute the statistic for. Defaults to 0.
        studentize (bool, optional): bool for studentize or not. Defaults to False.
        eps_param (int, optional): privacy parameter for parameter estimator. Defaults to None.
        eps_sigma (int, optional): privacy parameter for std dev estimator. Defaults to None.
        est (function, optional): estimator for the parameter. Defaults to mean1D.
        std_est (function, optional): estimator for the standard deviation. Defaults to std_mean1D.
        priv_est (function, optional): private estimator for the parameter. Defaults to None.
        priv_std_est (function, optional): private estimator for the standard deviation. Defaults to None.
        alpha (float, optional): confidence level. Defaults to 0.05.
    Returns:
        Box1D: Box1D object containing the lower and upper bounds of the confidence interval
    """

    stats_lst = bootstrap_stats_alldim(
        data=data,
        n_samples=n_samples,
        n_runs=n_runs,
        studentize=studentize,
        eps_sigma=eps_sigma,
        eps_param=eps_param,
        est=est,
        std_est=std_est,
        priv_est=priv_est,
        priv_std_est=priv_std_est,
    )
    out_lst = []

    tru_d = stats_lst.shape[1]
    for dim_val in range(tru_d):
        stats_lst_dim = stats_lst[:, dim_val]
        l_u = np.quantile(
            stats_lst_dim, [alpha / 2, 1 - alpha / 2], method="inverted_cdf"
        )

        out_lst.append(Box1D(l=l_u[0], u=l_u[1]))

    return out_lst


def blb_hists_nonpriv_alldim(
    data,
    H_all_dims,
    b,
    runs_per_boot,
    n_reps=None,
    samples_per_rep=None,
    eps_sigma=None,
    eps_param=None,
    studentize=False,
    est=mean1D,
    std_est=None,
    priv_est=None,
    priv_std_est=None,
    agg_alg="median",
    round_up_s=True,
):
    """[Only for regression problems multi dim]. This functions calculates the histograms  used in the bag of
    little bootstrap based algorithm to uniformly estimate the coverage of all
    sets in the histogram H
    Complexity: n_reps*samples_per_rep*(runs_per_boot*n + b)
    default: 1*n/b*(runs_per_boot*n + b)

    Args:
        data (array_like): input data
        H (Histogram): histogram object (Box1D type)
        b (int): size of each subset
        runs_per_boot (int): number of runs per bootstrap
        dim (int, optional): dimension to compute the statistic for. Defaults to 0.
        n_reps (int, optional): number of repetitions of the bag of little bootstrap. Defaults to None.
        samples_per_rep (int, optional): number of samples per repetition of the bag of little bootstrap. Defaults to None.
        eps_sigma (float, optional): privacy parameter for std dev estimator. Defaults to None.
        eps_param (float, optional): privacy parameter for parameter estimator. Defaults to None.
        studentize (bool, optional): bool for studentize or not. Defaults to False.
        est (function, optional): estimator for the parameter. Defaults to mean1D.
        std_est (function, optional): estimator for the standard deviation. Defaults to std_mean1D.
        priv_est (function, optional): private estimator for the parameter. Defaults to None.
        priv_std_est (function, optional): private estimator for the standard deviation. Defaults to None.

    Returns:
        list: if agg_alg == mean,
        A list of numpy arrays consisting of coverage probabilities of Hadamard histogram
        elif agg_alg == median,
        A list of a list of numpy arrays consisting of coverage probabilities of Hadamard histogram
    """

    n, d = data.shape

    # removing y dimension and adding intercept dimension
    # Only works for regression problems.
    d = d - 1 + 1

    assert agg_alg == "median"
    if n_reps is None:
        n_reps = 1

    s = int(n / b)
    if round_up_s and s * b < n - b / 2:
        s += 1

    for r in range(n_reps):
        # shouldn't you divide dataset into n_reps parts here
        perm = np.random.permutation(n)
        # returns a permutation of X
        data_perm = data[perm]

        if samples_per_rep is None:
            sample_idx = np.arange(s)
        else:
            assert samples_per_rep <= s
            sample_idx = np.random.choice(s, size=samples_per_rep, replace=False)
            # useful if want to use fewer chunks of data instead of all

        hist_dim_list = [[copy.deepcopy(H) for i in sample_idx] for H in H_all_dims]

        for i in sample_idx:
            if i == s - 1:
                data_b = data_perm[i * b :]
            else:
                data_b = data_perm[i * b : (i + 1) * b]

            stats_ot = bootstrap_stats_alldim(
                data=data_b,
                n_samples=n,
                n_runs=runs_per_boot,
                studentize=studentize,
                eps_sigma=eps_sigma,
                eps_param=eps_param,
                est=est,
                std_est=std_est,
                priv_est=priv_est,
                priv_std_est=priv_std_est,
            )

            assert stats_ot.shape[1] == d

            for dim_val in range(d):
                if "IntervalDiv" in str(type(hist_dim_list[dim_val][i])):
                    hist_dim_list[dim_val][i].add_batch(stats_ot[:, dim_val])
                else:
                    for stat in stats_ot[:, dim_val]:
                        hist_dim_list[dim_val][i].add(stat)

    return [[hist.get_histogram() for hist in hist_list] for hist_list in hist_dim_list]


def abthresh_blb_sens_alldim(
    data,
    H_all_dims,
    b,
    runs_per_boot,
    n_reps=None,
    samples_per_rep=None,
    eps_cdf=None,
    eps_sigma=None,
    eps_param=None,
    studentize=False,
    est=mean1D,
    std_est=None,
    priv_est=None,
    priv_std_est=None,
    oracle_min_length_lst=None,
    alpha=0.05,
    set_type="symm",
    err_correction=0,
):
    """[Only for regression problems multi dim] This functions performs above threshold by estimating the coverage of diferent sets using
    the bag of little bootstrap and aggregates histograms using the median (modified inverse sensitivity)
    Args:
        data (array_like): input data
        H (Histogram): histogram object (Hadamard1D type)
        b (int): size of each subset
        runs_per_boot (int): number of runs per bootstrap
        dim (int, optional): dimension to compute the statistic for. Defaults to 0.
        n_reps (int, optional): number of repetitions of the bag of little bootstrap. Defaults to None.
        samples_per_rep (int, optional): number of samples per repetition of the bag of little bootstrap. Defaults to None.
        eps_cdf (float, optional): privacy parameter for cdf estimator. Defaults to None.
        eps_sigma (float, optional): privacy parameter for std dev estimator. Defaults to None.
        eps_param (float, optional): privacy parameter for parameter estimator. Defaults to None.
        studentize (bool, optional): bool for studentize or not. Defaults to False.
        est (function, optional): estimator for the parameter. Defaults to mean1D.
        std_est (function, optional): estimator for the standard deviation. Defaults to std_mean1D.
        priv_est (function, optional): private estimator for the parameter. Defaults to None.
        priv_std_est (function, optional): private estimator for the standard deviation. Defaults to None.
        oracle_min_length (int, optional): minimum length of the confidence set provided by the oracle. Defaults to 0.
        alpha (float, optional): confidence level. Defaults to 0.05.
        set_type (bool, optional): if the confidence set is symmetric. Defaults to False.
        err_correction (int, optional): error correction value. Defaults to 0.

    Returns:
        Box1D: Private estimate of the confidence set
    """
    hist_dim_list = blb_hists_nonpriv_alldim(
        data=data,
        H_all_dims=H_all_dims,
        b=b,
        runs_per_boot=runs_per_boot,
        n_reps=n_reps,
        samples_per_rep=samples_per_rep,
        eps_sigma=eps_sigma,
        eps_param=eps_param,
        studentize=studentize,
        est=est,
        std_est=std_est,
        priv_est=priv_est,
        priv_std_est=priv_std_est,
        agg_alg="median",
    )

    global_eps = eps_cdf / 2
    local_eps = eps_cdf / 4
    int_lst = []
    total_dims = len(H_all_dims)
    dim_val = 0
    if oracle_min_length_lst is None:
        oracle_min_length_lst = [0] * total_dims

    for hist_list, H in zip(hist_dim_list, H_all_dims):
        num_int = max(int(oracle_min_length_lst[dim_val] / H.precision), 0)
        l_found = H.sets[H.depth - 1][0].l
        u_found = H.sets[H.depth - 1][-1].u
        broken = False
        if set_type == "symm":
            # Find a symmeteric set using above thresh
            num_int = num_int // 2
            tot_len = len(H.sets[H.depth - 1])
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
                l = H.sets[H.depth - 1][l_ind - i].l
                u = H.sets[H.depth - 1][u_ind + i].u
                A = Box1D(l, u)

                # generate list of coverage probability predictions
                cov_predict_list = cov_predict_batch(A, H, hist_list)
                if (
                    mod_inv_sens(cov_predict_list, global_eps, local_eps)
                    > 1 - alpha + err_correction
                ):
                    l_found = l
                    u_found = u
                    broken = True
                    break

        elif set_type == "short":
            while num_int < len(H.sets[H.depth - 1]):
                index_lst = list(range(num_int, len(H.sets[H.depth - 1])))
                iter_order = []
                q, r = divmod(len(index_lst), 2)
                if r == 0:
                    for i in range(q):
                        iter_order.append(index_lst[q - i])
                        iter_order.append(index_lst[q + i])
                else:
                    iter_order.append(index_lst[q])
                    for i in range(1, q + 1):
                        iter_order.append(index_lst[q - i])
                        iter_order.append(index_lst[q + i])
                # iter_order = index_lst
                for index in iter_order:
                    l = H.sets[H.depth - 1][index - num_int].l
                    u = H.sets[H.depth - 1][index].u

                    A = Box1D(l, u)
                    # generate list of coverage probability predictions
                    cov_predict_list = cov_predict_batch(A, H, hist_list)

                    if (
                        mod_inv_sens(cov_predict_list, global_eps, local_eps)
                        > 1 - alpha + err_correction
                    ):
                        l_found = l
                        u_found = u
                        broken = True
                        break

                if broken == True:
                    break
                num_int += 1
        else:
            raise ValueError("set_type not recognized")

        int_lst.append(Box1D(l_found, u_found))

    return int_lst


def fast_abthresh_blb_sens_alldim(
    data,
    H_all_dims,
    b,
    runs_per_boot,
    n_reps=None,
    samples_per_rep=None,
    eps_cdf=None,
    eps_sigma=None,
    eps_param=None,
    studentize=False,
    est=mean1D,
    std_est=None,
    priv_est=None,
    priv_std_est=None,
    oracle_min_length_lst=None,
    alpha=0.05,
    set_type="symm",
    err_correction=0,
):
    """[Only for regression problems multi dim] This functions performs above threshold by estimating the coverage of diferent sets using
    the bag of little bootstrap and aggregates histograms using the median (modified inverse sensitivity)
    Args:
        data (array_like): input data
        H (Histogram): histogram object (IntrvalDiv type)
        b (int): size of each subset
        runs_per_boot (int): number of runs per bootstrap
        dim (int, optional): dimension to compute the statistic for. Defaults to 0.
        n_reps (int, optional): number of repetitions of the bag of little bootstrap. Defaults to None.
        samples_per_rep (int, optional): number of samples per repetition of the bag of little bootstrap. Defaults to None.
        eps_cdf (float, optional): privacy parameter for cdf estimator. Defaults to None.
        eps_sigma (float, optional): privacy parameter for std dev estimator. Defaults to None.
        eps_param (float, optional): privacy parameter for parameter estimator. Defaults to None.
        studentize (bool, optional): bool for studentize or not. Defaults to False.
        est (function, optional): estimator for the parameter. Defaults to mean1D.
        std_est (function, optional): estimator for the standard deviation. Defaults to std_mean1D.
        priv_est (function, optional): private estimator for the parameter. Defaults to None.
        priv_std_est (function, optional): private estimator for the standard deviation. Defaults to None.
        oracle_min_length (int, optional): minimum length of the confidence set provided by the oracle. Defaults to 0.
        alpha (float, optional): confidence level. Defaults to 0.05.
        set_type (bool, optional): if the confidence set is symmetric. Defaults to False.
        err_correction (int, optional): error correction value. Defaults to 0.

    Returns:
        Box1D: Private estimate of the confidence set
    """
    hist_dim_list = blb_hists_nonpriv_alldim(
        data=data,
        H_all_dims=H_all_dims,
        b=b,
        runs_per_boot=runs_per_boot,
        n_reps=n_reps,
        samples_per_rep=samples_per_rep,
        eps_sigma=eps_sigma,
        eps_param=eps_param,
        studentize=studentize,
        est=est,
        std_est=std_est,
        priv_est=priv_est,
        priv_std_est=priv_std_est,
        agg_alg="median",
    )

    global_eps = eps_cdf / 2
    local_eps = eps_cdf / 4

    if oracle_min_length_lst is None:
        oracle_min_length_lst = [0] * len(H_all_dims)

    dim_val = 0
    int_lst = []
    for hist_list, H in zip(hist_dim_list, H_all_dims):
        num_int = max(int(oracle_min_length_lst[dim_val] / H.precision), 0)
        l_found = H.l_arr[0]
        u_found = H.u_arr[-1]
        broken = False

        tot_len = len(H.l_arr)
        if set_type == "symm":
            # Find a symmeteric set using above thresh
            num_int = num_int // 2
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
                l = H.l_arr[l_ind - i]
                u = H.u_arr[u_ind + i]
                A = Box1D(l, u)

                # generate list of coverage probability predictions
                cov_predict_list = cov_predict_int_batch(A, H, hist_list)

                if (
                    mod_inv_sens(cov_predict_list, global_eps, local_eps)
                    > 1 - alpha + err_correction
                ):
                    l_found = l
                    u_found = u
                    broken = True
                    break
        elif set_type == "short":
            while num_int < tot_len:
                index_lst = list(range(num_int, tot_len))
                iter_order = []
                q, r = divmod(len(index_lst), 2)
                if r == 0:
                    for i in range(q):
                        iter_order.append(index_lst[q - i])
                        iter_order.append(index_lst[q + i])
                else:
                    iter_order.append(index_lst[q])
                    for i in range(1, q + 1):
                        iter_order.append(index_lst[q - i])
                        iter_order.append(index_lst[q + i])
                # iter_order = index_lst
                for index in iter_order:
                    l = H.l_arr[index - num_int]
                    u = H.u_arr[index]

                    A = Box1D(l, u)
                    # generate list of coverage probability predictions
                    cov_predict_list = cov_predict_int_batch(A, H, hist_list)

                    if (
                        mod_inv_sens(cov_predict_list, global_eps, local_eps)
                        > 1 - alpha + err_correction
                    ):
                        l_found = l
                        u_found = u
                        broken = True
                        break

                if broken == True:
                    break
                num_int += 1
        else:
            raise ValueError("set_type not recognized")

        int_lst.append(Box1D(l_found, u_found))

    return int_lst
