### Common PRIVATE MECHANISMS ###
import numpy as np
import copy


def lap(x, eps, L1_sen):
    """Laplace mechanism

    Args:
        x (array): true estimator (generally 1D)
        eps (float):  privacy parameter
        L1_sen (_type_): L1 sensitivity (maybe a vector)

    Returns:
        same as input: privatized estimator by adding laplace noise
    """
    b = L1_sen / eps
    return x + np.random.laplace(loc=0.0, scale=b, size=x.shape)


def priv_med_gumbel(X, eps, rho_granularity=0, R_low=0, R=1, sampling_type="gumbel"):
    """Outputs the private median using inverse sensitivity mechanism
    Assumes the data lies in [R_low,R]

    Args:
        X (1D array): data
        eps (float): privacy parameter epsilon
        R (int, optional): Upper bound on data, Defaults to 1.

    Returns:
        float: private median estimate of X
    """
    if eps == np.inf:
        return np.median(X)
    assert X.ndim == 1
    # calculate the median and offset and sort the array
    X = np.array(X)
    X.sort()
    try:
        assert R_low <= X[0] and X[-1] <= R
    except AssertionError:
        print(f"Data lies outside [{R_low},{R}], X[0]={X[0]}, X[-1]={X[-1]}")
    # m, offset = np.median(X), (len(X) % 2) - 1
    n = len(X)
    scrs = np.zeros(n + 1)
    if n % 2 == 0:
        if rho_granularity > 0:
            X[: n // 2] -= rho_granularity
            X[n // 2 :] += rho_granularity
            X = np.clip(X, R_low, R)
        if X[0] == R_low:
            scrs[0] = -np.inf
        else:
            scrs[0] = np.log(X[0] - R_low) - n // 2 * eps / 2
        scrs[1:n] = np.log(X[1:] - X[:-1]) - np.abs(n // 2 - np.arange(1, n)) * eps / 2
        scrs[n] = np.log(R - X[-1]) - n // 2 * eps / 2
    else:
        if rho_granularity > 0:
            X[: n // 2] -= rho_granularity
            X[n // 2 + 1 :] += rho_granularity
            X = np.clip(X, R_low, R)
        if X[0] == R_low:
            scrs[0] = -np.inf
        else:
            scrs[0] = np.log(X[0] - R_low) - (n + 1) // 2 * eps / 2
        scrs[1 : (n + 1) // 2] = (
            np.log(X[1 : (n + 1) // 2] - X[: (n - 1) // 2])
            - np.abs((n + 1) // 2 - np.arange(1, (n + 1) // 2)) * eps / 2
        )
        scrs[(n + 1) // 2 : n] = (
            np.log(X[(n + 1) // 2 :] - X[(n - 1) // 2 : -1])
            - np.abs((n - 1) // 2 - np.arange((n + 1) // 2, n)) * eps / 2
        )
        scrs[n] = np.log(R - X[-1]) - (n + 1) // 2 * eps / 2

    assert len(scrs) == n + 1
    scrs_np = np.array(scrs)
    if sampling_type == "gumbel":
        noisy_scrs = scrs_np + np.random.gumbel(loc=0.0, scale=1.0, size=scrs_np.shape)
        samp_idx = np.argmax(noisy_scrs)
    elif sampling_type == "softmax":
        softmax_scrs = np.exp(scrs_np) / np.sum(np.exp(scrs_np))
        samp_idx = np.random.choice(n + 1, p=softmax_scrs)
    else:
        raise ValueError(f"Sampling type {sampling_type} not supported")
    if samp_idx == 0:
        return np.random.uniform(low=R_low, high=X[0])
    elif samp_idx == n:
        return np.random.uniform(low=X[-1], high=R)
    else:
        return np.random.uniform(low=X[samp_idx - 1], high=X[samp_idx])


def mod_inv_sens(X, global_eps, local_eps, R=1):
    """Implements the modified inverse sensitivity mechanism
     to be used in conjunction with above thresh to calculate confidence sets.

    Args:
        X (array_like): data
        global_eps (_type_): global noise parameter
        local_eps (_type_): local noise parameter
        R (int, optional): we assume data lies in [0,R]. Defaults to 1.
    """
    if global_eps == np.inf:
        return np.median(X)
    X = np.array(X)
    X.sort()
    n = np.shape(X)[0]

    global_noise = np.random.laplace(loc=n / 2, scale=1 / global_eps)
    local_noise = np.random.laplace(loc=0, scale=1 / local_eps)

    idx = int(global_noise + local_noise)

    if idx < 1:
        return 0
    elif idx > n:
        return R
    else:
        # correcting for zero indexing in python
        return X[idx - 1]


def in_set(x, low, high):
    """
    returns 1 if x is in [low,high]

    Args:
        x (float): query
        low (float): lower end of interval
        high (float): higher end of interval

    Returns:
        bool: 1 if x is in [low,high]
    """
    if x >= low and x <= high:
        return 1
    else:
        return 0
