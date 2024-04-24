import numpy as np
from scipy.linalg import sqrtm
from dp_inference.priv_mech import lap, priv_med_gumbel
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fmin_l_bfgs_b

from scipy.special import expit
from scipy.stats import gamma
from scipy.special import gammainc


# Bounded and variance estimators
def mean(X):
    return np.mean(X, axis=0)


def mean1D(X):
    return np.mean(X)


def std_mean(data):
    X = data
    n, d = X.shape
    return sqrtm(1 / (n - 1) * (X - mean(X)).T @ (X - mean(X)))


def std_mean1D(X):
    n = X.shape[0]
    X = X[:, 0]
    mu = np.mean(X)
    return np.sqrt(1 / (n - 1) * (X - mu).T @ (X - mu))


def priv_mean1D(X, trunc_low, trunc_high, epsilon):
    # for bounded random variables
    n = X.shape[0]
    trunc_samp_mean = np.mean(X)
    return lap(trunc_samp_mean, epsilon, (trunc_high - trunc_low) / n)


def priv_std_mean1D(data, trunc_low, trunc_high, epsilon):
    # Naive implementation of private standard deviation
    # for bounded random variables in 1D
    X = data[:, 0]
    n = X.shape[0]
    max_abs = max(np.abs(trunc_low), np.abs(trunc_high))
    l1_sens = max_abs**2 / n + 2 * max_abs * (trunc_high - trunc_low) / (n * (n - 1))
    return lap(std_mean1D(data), epsilon, l1_sens)


# Median estimators
def median(X):
    return np.median(X, axis=0)


# Logistic regression estimators
def sigmoid(x):
    return expit(x)


def logreg_obj(beta, X, y):
    n, d = X.shape
    exponent = np.multiply(y, X @ beta)
    return 1 / n * np.sum(np.log(1 + np.exp(-exponent)))


def logreg_grad(beta, X, y):
    n, d = X.shape
    exponent = np.multiply(y, X @ beta)
    return (
        1
        / n
        * np.sum(
            -np.multiply(np.tile(np.multiply(sigmoid(-exponent), y), [d, 1]), X.T),
            axis=1,
        )
    )


def logreg_hess(beta, X, y):
    n, d = X.shape
    exponent = np.multiply(y, X @ beta)
    weights = np.multiply(sigmoid(exponent), 1 - sigmoid(exponent))
    return 1 / n * np.multiply(np.tile(weights, [d, 1]), X.T) @ X


def hess_inv_square(data, intercept=True):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape
    if intercept:
        d = d + 1
        X = np.hstack((X, np.ones((n, 1))))
    if intercept:
        beta = logreg_est_intercept(data, alg="newton", err_tol=1e-6)
    else:
        beta = logreg_est(data, alg="newton", err_tol=1e-6)

    hess_inv = np.linalg.inv(logreg_hess(beta, X, y) + 1e-10 * np.eye(d))

    return np.matmul(hess_inv, hess_inv.T)


def logreg_std(data, intercept=True):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape
    if intercept:
        beta = logreg_est_intercept(data, alg="newton", err_tol=1e-6)
    else:
        beta = logreg_est(data, alg="newton", err_tol=1e-6)

    if intercept:
        # add intercept
        X = np.hstack((X, np.ones((n, 1))))
        d += 1

    hess_inv = np.linalg.inv(logreg_hess(beta, X, y) + 1e-10 * np.eye(d))
    exponent = np.multiply(y, X @ beta)
    grads = -np.multiply(np.tile(np.multiply(sigmoid(-exponent), y), [d, 1]), X.T)
    grad_cov = grads @ grads.T / n
    # print(grad_cov)
    # print(hess_inv)
    return sqrtm(hess_inv @ grad_cov @ hess_inv)


def logreg_est(
    data,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-3,
):
    X = data[:, 0:-1]
    y = data[:, -1]

    n, d = X.shape
    delta = 1e-6
    if alg == "newton":
        # NEWTON'S METHOD
        if f0 is None:
            f0 = np.zeros(d)
        beta = f0
        curr_err = 1
        n_iter = 0
        while curr_err > err_tol and n_iter < 100:
            grad = logreg_grad(beta, X, y)
            hess = logreg_hess(beta, X, y)
            beta_new = beta - np.linalg.inv(hess + delta * np.eye(d)) @ grad
            curr_err = np.linalg.norm(beta_new - beta)
            beta = beta_new
            n_iter += 1
    elif alg == "fmin_lbfgs":
        # L-BFGS-B from scipy
        if f0 is None:
            f0 = np.zeros(d)
        beta, opt_val, d = fmin_l_bfgs_b(
            logreg_obj,
            f0,
            fprime=logreg_grad,
            args=(X, y),
        )
    elif alg == "sklearn":
        # sklearn logistic regression different solvers
        logreg = LogisticRegression(
            solver=logreg_solver,
            fit_intercept=False,
            penalty="none",
            warm_start=warm_start,
        )
        logreg.fit(X, y)
        beta = logreg.coef_[0]
    else:
        raise ValueError("Invalid algorithm specified")

    return beta


def logreg_est_intercept(
    data,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-3,
):
    X = data[:, 0:-1]
    y = data[:, -1]

    n, d = X.shape
    delta = 1e-6
    if alg == "newton":
        # NEWTON'S METHOD

        # add intercept
        X = np.hstack((X, np.ones((n, 1))))
        if f0 is None:
            f0 = np.zeros(d + 1)
        beta = f0
        curr_err = 1
        n_iter = 0
        while curr_err > err_tol and n_iter < 100:
            grad = logreg_grad(beta, X, y)
            hess = logreg_hess(beta, X, y)
            beta_new = beta - np.linalg.inv(hess + delta * np.eye(d + 1)) @ grad
            curr_err = np.linalg.norm(beta_new - beta)
            # print("Iteration {}, error {}".format(n_iter, curr_err))
            beta = beta_new
            n_iter += 1
        beta = beta_new[:-1]
        intercept = beta_new[-1]
        # print(n_iter)
    elif alg == "fmin_lbfgs":
        # L-BFGS-B from scipy

        # add intercept
        X = np.hstack((X, np.ones((n, 1))))
        if f0 is None:
            f0 = np.zeros(d + 1)
        beta_scp, opt_val, d = fmin_l_bfgs_b(
            logreg_obj,
            f0,
            fprime=logreg_grad,
            args=(X, y),
        )
        beta = beta_scp[:-1]
        intercept = beta_scp[-1]
    elif alg == "sklearn":
        # sklearn logistic regression different solvers
        logreg = LogisticRegression(
            solver=logreg_solver,
            fit_intercept=True,
            penalty="none",
            warm_start=warm_start,
            n_jobs=1,
        )
        logreg.fit(X, y)
        beta = logreg.coef_[0]
        intercept = logreg.intercept_[0]
    else:
        raise ValueError("Invalid algorithm specified")

    decision_vector = np.concatenate([beta, np.array([intercept])])

    return decision_vector


def gen_logreg_data(n, d, beta, dist="uniform", intercept=0):
    if len(beta) == d:
        beta = np.concatenate([beta, np.array([intercept])])

    if dist == "uniform":
        X = np.random.uniform(-1, 1, [n, d])
    elif dist == "normal":
        X = np.random.normal(0, 1, [n, d])
    y = np.random.binomial(1, sigmoid(X @ beta[:-1] + intercept))
    y = 2 * y - 1

    return np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)


def proj_SGD_logreg(data, step, dom_cen, dom_diam, beta0):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape
    beta = beta0
    beta_lst = []
    curr_step = step
    log_iter = 1
    for i in range(n):
        beta = beta - curr_step * (
            logreg_grad(beta, X[i].reshape(1, -1), y[i].reshape(1, -1))
        )
        beta = proj_l2_ball(beta, dom_cen, dom_diam)
        beta_lst.append(beta)
        if i == n - np.ceil(n / 2**log_iter):
            curr_step = curr_step / 2
            log_iter += 1

    return np.mean(beta_lst, axis=0)


def proj_l2_ball(beta, dom_cen, dom_diam):
    if np.linalg.norm(beta - dom_cen) > dom_diam:
        return dom_cen + (beta - dom_cen) / np.linalg.norm(beta - dom_cen) * dom_diam
    else:
        return beta


def priv_logreg_inv_sens_est(
    data,
    epsilon=np.inf,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-5,
    norm_bound=5,
    nonpriv_est=None,
):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape

    if nonpriv_est == None:
        beta_opt = logreg_est(
            data,
            alg=alg,
            logreg_solver=logreg_solver,
            warm_start=warm_start,
            f0=f0,
            err_tol=err_tol,
        )
    else:
        beta_opt = nonpriv_est.copy()

    radius = gamma.rvs(
        a=d,
    )
    x = np.random.standard_normal(d)
    uni_ran_vec = x / np.linalg.norm(x)
    hess_tru = logreg_hess(beta_opt, X, y)
    hess_inv = np.linalg.inv(hess_tru + 10**-7 * np.eye(d))
    noise_vec = (radius * 2 * norm_bound / (n * epsilon)) * hess_inv @ uni_ran_vec

    beta_opt += noise_vec

    return beta_opt


def priv_intercept_logreg_inv_sens_est(
    data,
    epsilon=np.inf,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-5,
    norm_bound=5,
    nonpriv_est=None,
):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape

    if nonpriv_est is None:
        decision_vector_opt = logreg_est_intercept(
            data,
            alg=alg,
            logreg_solver=logreg_solver,
            warm_start=warm_start,
            f0=f0,
            err_tol=err_tol,
        )
    else:
        decision_vector_opt = nonpriv_est.copy()

    d = len(decision_vector_opt)
    # add intercept
    X = np.hstack((X, np.ones((n, 1))))
    radius = gamma.rvs(
        a=d,
    )
    x = np.random.standard_normal(d)
    uni_ran_vec = x / np.linalg.norm(x)
    hess_tru = logreg_hess(decision_vector_opt, X, y)
    hess_inv = np.linalg.inv(hess_tru + 10**-7 * np.eye(d))
    noise_vec = (radius * 2 * norm_bound / (n * epsilon)) * hess_inv @ uni_ran_vec

    decision_vector_opt += noise_vec

    return decision_vector_opt


def priv_logreg_invsens_mh(
    data,
    epsilon=np.inf,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-3,
    norm_bound=5,
    mh_rounds=100,
    dom_diam=5,
    dom_cen=None,
    rn_exp=-5 / 6,
    rn_const=4,
):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape
    if not dom_cen:
        dom_cen = np.zeros(d)

    rn = rn_const * n**rn_exp
    # theta_n from the paper
    beta_opt = logreg_est(
        data,
        alg=alg,
        logreg_solver=logreg_solver,
        warm_start=warm_start,
        f0=f0,
        err_tol=err_tol,
    )
    radius = gamma.rvs(
        a=d,
    )
    hess_tru = logreg_hess(beta_opt, X, y)
    hess_inv = np.linalg.inv(hess_tru + 10**-7 * np.eye(d))

    r_calc = epsilon / (2 * norm_bound) * n * rn
    A = hess_tru * n * epsilon / (2 * norm_bound)

    alpha = (
        (np.linalg.det(A) * dom_diam**d - r_calc**d)
        * np.exp(-r_calc)
        / (
            np.exp(-r_calc) * (np.linalg.det(A) * dom_diam**d - r_calc**d)
            + gammainc(d, r_calc)
        )
    )
    # print(f"alpha: {alpha}")
    # print(f"gammainc: {gammainc(d, r_calc)}")
    # print(f"det(A): {np.linalg.det(A)}")
    # print(f"r_calc: {r_calc}")
    # print(f"dom_diam**d: {dom_diam**d}")
    # print(f"r_calc**d: {r_calc**d}")
    # print(f"exp(-r_calc): {np.exp(-r_calc)}")

    try:
        assert np.linalg.norm(hess_tru, ord=-2) * dom_diam - rn > 0
    except AssertionError:  # pragma: no cover
        print(f"dom_diam: {dom_diam}")
        print(f"rn: {rn}")
        print(f"norm(hess): {np.linalg.norm(hess_tru, ord=-2)}")
        print(
            f"norm(hess) * dom_diam - rn: {np.linalg.norm(hess_tru, ord=-2) * dom_diam - rn}"
        )
        raise
    beta_curr = beta_opt

    grad_curr = logreg_grad(beta_curr, X, y)

    q_curr = 1
    switch = 0
    for i in range(mh_rounds):
        rejected_num = 0
        coin_flip = np.random.binomial(1, alpha)
        if coin_flip:
            out_of_domain = True
            while out_of_domain:
                x = np.random.standard_normal(d)
                uni_ran_vec = x / np.linalg.norm(x)
                radius = np.random.uniform(0, dom_diam)
                noise_vec = radius * uni_ran_vec
                if np.linalg.norm(hess_tru * noise_vec) > rn:
                    beta_candidate = beta_opt + noise_vec
                    out_of_domain = False
                rejected_num += 1
                if rejected_num % 1000 == 0:
                    print(f"rejected {rejected_num} coin_flip {coin_flip}")
                    print(
                        f"hess_noise_norm: {np.linalg.norm(hess_tru * noise_vec)}, rn: {rn}, alpha: {alpha}"
                    )

        else:
            out_of_domain = True
            while out_of_domain:
                rejected_num += 1
                if rejected_num > 1000:
                    print(f"rejected {rejected_num} coin_flip {coin_flip}")
                    print(
                        f"dom_diam: {dom_diam} domain: {np.linalg.norm(beta_candidate - dom_cen)}, rn: {rn}, hess_vec_norm: {hess_vec_norm}"
                    )
                    print(alpha)
                    raise ValueError("not enough domain diameter")
                radius = gamma.rvs(
                    a=d,
                )
                x = np.random.standard_normal(d)
                uni_ran_vec = x / np.linalg.norm(x)
                noise_vec = (
                    (radius * 2 * norm_bound / (n * epsilon)) * hess_inv @ uni_ran_vec
                )

                beta_candidate = beta_opt + noise_vec

                if np.linalg.norm(beta_candidate - dom_cen) > dom_diam:
                    continue

                hess_vec_norm = np.linalg.norm(hess_tru @ noise_vec)

                if hess_vec_norm <= rn:
                    out_of_domain = False

        # print(f"rejected {rejected_num} coin_flip {coin_flip}")

        grad_candidate = logreg_grad(beta_candidate, X, y)
        pi_ratio = np.exp(
            -0.5
            * epsilon
            * (
                np.ceil(np.linalg.norm(grad_candidate) * n / norm_bound)
                - np.ceil(np.linalg.norm(grad_curr) * n / norm_bound)
            )
        )
        q_candidate = np.exp(-min(np.linalg.norm(A @ noise_vec), r_calc))

        prob = min(pi_ratio * q_curr / (q_candidate), 1)
        bin_sample = np.random.binomial(1, prob)
        if bin_sample == 1:
            switch += 1
            beta_curr = beta_candidate
            grad_curr = grad_candidate
            q_curr = q_candidate

    # print(
    #     f"switch {switch} prob {prob} pi_ratio {pi_ratio} q_curr {q_curr} q_candidate {q_candidate}"
    # )
    return beta_curr


def priv_intercept_logreg_invsens_mh(
    data,
    epsilon=np.inf,
    alg="newton",
    logreg_solver="lbfgs",
    warm_start=False,
    f0=None,
    err_tol=1e-3,
    norm_bound=5,
    mh_rounds=100,
    dom_diam=5,
    dom_cen=None,
    rn_exp=-5 / 6,
    rn_const=4,
):
    X = data[:, 0:-1]
    y = data[:, -1]
    n, d = X.shape
    if not dom_cen:
        dom_cen = np.zeros(d)

    rn = rn_const * n**rn_exp
    # theta_n from the paper
    beta_opt = logreg_est_intercept(
        data,
        alg=alg,
        logreg_solver=logreg_solver,
        warm_start=warm_start,
        f0=f0,
        err_tol=err_tol,
    )
    d = len(beta_opt)
    # add intercept
    X = np.hstack((np.ones((n, 1)), X))

    radius = gamma.rvs(
        a=d,
    )
    hess_tru = logreg_hess(beta_opt, X, y)
    hess_inv = np.linalg.inv(hess_tru + 10**-7 * np.eye(d))

    r_calc = epsilon / (2 * norm_bound) * n * rn
    A = hess_tru * n * epsilon / (2 * norm_bound)

    alpha = (
        (np.linalg.det(A) * dom_diam**d - r_calc**d)
        * np.exp(-r_calc)
        / (
            np.exp(-r_calc) * (np.linalg.det(A) * dom_diam**d - r_calc**d)
            + gammainc(d, r_calc)
        )
    )

    try:
        assert np.linalg.norm(hess_tru, ord=-2) * dom_diam - rn > 0
    except AssertionError:  # pragma: no cover
        print(f"dom_diam: {dom_diam}")
        print(f"rn: {rn}")
        print(f"norm(hess): {np.linalg.norm(hess_tru, ord=-2)}")
        print(
            f"norm(hess) * dom_diam - rn: {np.linalg.norm(hess_tru, ord=-2) * dom_diam - rn}"
        )
        raise
    beta_curr = beta_opt

    grad_curr = logreg_grad(beta_curr, X, y)

    q_curr = 1
    switch = 0
    for i in range(mh_rounds):
        rejected_num = 0
        coin_flip = np.random.binomial(1, alpha)
        if coin_flip:
            out_of_domain = True
            while out_of_domain:
                x = np.random.standard_normal(d)
                uni_ran_vec = x / np.linalg.norm(x)
                radius = np.random.uniform(0, dom_diam)
                noise_vec = radius * uni_ran_vec
                if np.linalg.norm(hess_tru * noise_vec) > rn:
                    beta_candidate = beta_opt + noise_vec
                    out_of_domain = False
                rejected_num += 1
                if rejected_num % 1000 == 0:
                    print(f"rejected {rejected_num} coin_flip {coin_flip}")
                    print(
                        f"hess_noise_norm: {np.linalg.norm(hess_tru * noise_vec)}, rn: {rn}, alpha: {alpha}"
                    )

        else:
            out_of_domain = True
            while out_of_domain:
                rejected_num += 1
                if rejected_num > 1000:
                    print(f"rejected {rejected_num} coin_flip {coin_flip}")
                    print(
                        f"dom_diam: {dom_diam} domain: {np.linalg.norm(beta_candidate - dom_cen)}, rn: {rn}, hess_vec_norm: {hess_vec_norm}"
                    )
                    print(alpha)
                    raise ValueError("not enough domain diameter")
                radius = gamma.rvs(
                    a=d,
                )
                x = np.random.standard_normal(d)
                uni_ran_vec = x / np.linalg.norm(x)
                noise_vec = (
                    (radius * 2 * norm_bound / (n * epsilon)) * hess_inv @ uni_ran_vec
                )

                beta_candidate = beta_opt + noise_vec

                if np.linalg.norm(beta_candidate - dom_cen) > dom_diam:
                    continue

                hess_vec_norm = np.linalg.norm(hess_tru @ noise_vec)

                if hess_vec_norm <= rn:
                    out_of_domain = False

        # print(f"rejected {rejected_num} coin_flip {coin_flip}")

        grad_candidate = logreg_grad(beta_candidate, X, y)
        pi_ratio = np.exp(
            -0.5
            * epsilon
            * (
                np.ceil(np.linalg.norm(grad_candidate) * n / norm_bound)
                - np.ceil(np.linalg.norm(grad_curr) * n / norm_bound)
            )
        )
        q_candidate = np.exp(-min(np.linalg.norm(A @ noise_vec), r_calc))

        prob = min(pi_ratio * q_curr / (q_candidate), 1)
        bin_sample = np.random.binomial(1, prob)
        if bin_sample == 1:
            switch += 1
            beta_curr = beta_candidate
            grad_curr = grad_candidate
            q_curr = q_candidate

    # print(
    #     f"switch {switch} prob {prob} pi_ratio {pi_ratio} q_curr {q_curr} q_candidate {q_candidate}"
    # )
    return beta_curr


def priv_blb_std_est1D(
    data,
    b,
    est,
    runs_per_boot,
    eps_sigma,
    R_upp_var,
    round_up_s=True,
    rho_granularity=0,
    ablation_run=False,
):
    n, d = data.shape
    s = int(n / b)
    if round_up_s and s * b < n - 1:
        s += 1

    perm = np.random.permutation(n)
    # returns a permutation of X
    data_perm = data[perm]
    sample_idx = np.arange(s)

    lil_bs_out = []
    for i in sample_idx:
        if i == s - 1:
            data_b = data_perm[i * b :]
        else:
            data_b = data_perm[i * b : (i + 1) * b]
        if ablation_run:
            data_b = data_perm

        est_arr = []
        for j in range(runs_per_boot):
            resampled_idxs = np.random.choice(len(data_b), n)
            resampled_data = data_b[resampled_idxs]
            est_arr.append(est(resampled_data))

        lil_bs_out.append(np.cov(est_arr, rowvar=False))

    np_lil_bs = np.array(lil_bs_out)

    if eps_sigma == np.inf:
        return np.sqrt(np.median(np_lil_bs))
    else:
        np_lil_bs = np.clip(np.nan_to_num(np_lil_bs, nan=R_upp_var), 0, R_upp_var)
        return np.sqrt(
            priv_med_gumbel(
                np_lil_bs, eps_sigma, rho_granularity=rho_granularity, R=R_upp_var
            )
        )


def priv_blb_std_est_all_dim(
    data,
    b,
    est,
    runs_per_boot,
    eps_sigma,
    R_upp_var_lst,
    round_up_s=True,
    rho_granularity=0,
):
    n, d = data.shape

    # removing y dimension and adding intercept dimension
    d = d - 1 + 1

    s = int(n / b)
    if round_up_s and s * b < n - b / 2:
        s += 1
    perm = np.random.permutation(n)
    # returns a permutation of X
    data_perm = data[perm]
    sample_idx = np.arange(s)

    lil_bs_out = []
    for i in sample_idx:
        if i == s - 1:
            data_b = data_perm[i * b :]
        else:
            data_b = data_perm[i * b : (i + 1) * b]

        est_arr = []
        for j in range(runs_per_boot):
            resampled_idxs = np.random.choice(len(data_b), n)
            resampled_data = data_b[resampled_idxs]
            est_arr.append(est(resampled_data))

        # Only save diagonal
        lil_bs_out.append(np.diag(np.cov(est_arr, rowvar=False)))

    var_est_out = []
    np_lil_bs = np.array(lil_bs_out)

    for dim_val in range(np_lil_bs.shape[1]):
        np_lil_bs[:, dim_val] = np.clip(
            np.nan_to_num(np_lil_bs[:, dim_val], nan=R_upp_var_lst[dim_val]),
            0,
            R_upp_var_lst[dim_val],
        )
        var_est_out.append(
            priv_med_gumbel(
                np_lil_bs[:, dim_val],
                eps_sigma,
                rho_granularity=rho_granularity,
                R=R_upp_var_lst[dim_val],
            )
        )

    return np.sqrt(var_est_out)
