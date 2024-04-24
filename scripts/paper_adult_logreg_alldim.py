import os

import numpy as np

import pandas as pd
import os
from scipy.stats import t as t_dist

os.environ["MKL_NUM_THREADS"] = "1"

from dp_inference.common_dist_framework import IntervalDiv, Box1D
from dp_inference.inference_methods_all_dim import (
    fast_abthresh_blb_sens_alldim,
    bootstrap_quantile_ci_alldim,
)
from dp_inference.estimators import (
    logreg_est_intercept,
    priv_intercept_logreg_inv_sens_est,
    priv_intercept_logreg_invsens_mh,
    priv_blb_std_est_all_dim,
)
from dp_inference.priv_mech import in_set

import argparse
import time

import multiprocessing as mp
from datetime import date

method_lst = [
    "normal_proper_blb",
    "proper_unstud",
    "boot_unstud_priv",
]

set_type = "symm"  # "eqtail" or "symm"

const_std_up = 10
const_inv_sens = 10

min_boots = 100
max_boots = 10000

alpha = 0.05
mc_exp = 1.5
gran_exp = 0.5

num_trials = 240
num_resamp = 320
search_range_const = 5

####### PRIVACY PARAMETERS ########
eps_total_lst = [10, 8, 5]  # [10, 5, 2]
eps_param_frac_lst = [0.5]  # [0.5, 0.3, 0.1]

parser = argparse.ArgumentParser(description="Collecting the input")
parser.add_argument(
    "--dist",
    type=str,
    default="Adult",
    help="distribution",
    choices=[
        "Adult",
    ],
)

parser.add_argument(
    "--round",
    type=int,
    default=0,
)

args = parser.parse_args()

dist = args.dist
round_up = args.round
print(dist)

if round_up == 0:
    round_up_prec = False
else:
    round_up_prec = True
print(set_type)


#############################################
# Load data #
#############################################
data_dir = "data/preprocessed14/"
features = pd.read_csv(data_dir + "features.csv")
labels = pd.read_csv(data_dir + "labels.csv")

print("Loaded data")

active_feature_lst = []
active_feature_lst.extend(["AGEP"])
active_feature_lst.append("SCHL")
active_feature_lst.append("WKHP")
active_feature_lst.append("SEX")

norm_bound = 1
features = features[active_feature_lst]

logreg_d = features.shape[1]
print("logreg_d", logreg_d)
logreg_d_intercept = logreg_d + 1

# Estimators
alg = "newton"
err_tol = 1e-6
lr_est = lambda data, f0: logreg_est_intercept(
    data, alg=alg, logreg_solver="lbfgs", warm_start=True, f0=f0, err_tol=err_tol
)
priv_lr_est_approx = (
    lambda data, eps, f0, nonpriv_est=None: priv_intercept_logreg_inv_sens_est(
        data,
        eps,
        alg=alg,
        logreg_solver="lbfgs",
        warm_start=True,
        f0=f0,
        norm_bound=norm_bound,
        err_tol=err_tol,
        nonpriv_est=nonpriv_est,
    )
)
priv_lr_est_mh = (
    lambda data, eps, f0, dom_diam, nonpriv_est=None: priv_intercept_logreg_invsens_mh(
        data,
        eps,
        alg=alg,
        warm_start=True,
        f0=f0,
        dom_diam=dom_diam,
        norm_bound=norm_bound,
        err_tol=err_tol,
        nonpriv_est=nonpriv_est,
    )
)


########### Find global solution ############
full_data = np.concatenate([features.to_numpy(), labels.to_numpy()], axis=1)

start_time = time.time()
beta_tru = lr_est(full_data, f0=None)
print("Found global solution in time ", time.time() - start_time)

#################################################################


boot_run = True
alpha = 0.05
oracle_min_length_lst = [0] * logreg_d_intercept  # []
err_correction_unstud = lambda n: 0
multiplier_lst = [5, 5, 5, 5, 5]  # []
precision_unstud = lambda multiplier, n: multiplier * 2 / (n**0.5)
MAX_ACTUAL_RUNS = 1000000

eps_total = 8
eps_param_frac = 0.5
eps_param = eps_param_frac * eps_total
eps_cdf = eps_total - eps_param

search_range_const_lst = [70, 100, 120, 20, 100]  # []
const_inv_sens = 10

s_sens = lambda n: np.ceil(const_inv_sens * np.log(n) / eps_cdf)
b_sens_unstud = lambda n: int(n / s_sens(n))
runs_per_boot_sens_unstud = lambda n: int(
    min(
        n ** (mc_exp - 1) * b_sens_unstud(n) / (np.log(n)),
        MAX_ACTUAL_RUNS * b_sens_unstud(n) / n,
    )
)

n_boot = lambda n: int(min(n**mc_exp / (np.log(n)), MAX_ACTUAL_RUNS))


n_lst = [2000, 4000, 6000, 8000]

# n_lst = [1000]

#################################################################
##########Var Estimator########################

R_upp_var = lambda n: np.array(
    [
        (search_range_const_lst[i] / 4 * const_std_up) ** 2 / n
        for i in range(logreg_d_intercept)
    ]
)

priv_noise_n_std_est = lambda data, eps, f0: priv_blb_std_est_all_dim(
    data,
    b=b_sens_unstud(len(data)),
    est=lambda X: priv_lr_est_approx(X, eps_param, f0),
    runs_per_boot=runs_per_boot_sens_unstud(len(data)),
    eps_sigma=eps,
    R_upp_var_lst=R_upp_var(len(data)),
    rho_granularity=1 / len(data) ** (1 + gran_exp),
)


### Versioning folder name ###

dir_prefix = f"results/Final_adult_cov_{1 - alpha}_and_len/"
os.system("mkdir -p " + dir_prefix)

day = date.today().day
month = date.today().month
version = 0

file_list = [file for file in os.listdir(dir_prefix) if f"m{month}d{day}" in file]
version += len(file_list)

save_dir = f"{dir_prefix}m{month}d{day}v{version}/"

########### Attribute Dict #############
attr_dict = {
    "dist": dist,
    "alpha": alpha,
    "mc_exp": mc_exp,
    "gran_exp": gran_exp,
    "const_std_up": const_std_up,
    "const_inv_sens": const_inv_sens,
}

### EXPERIMENTS ###

columns = [
    "n",
    "trial",
    "resamp",
    "eps_total",
    "eps_param",
]
columns.extend(method_lst)


def parallel_trials(n, trial):
    global precision_unstud
    global dist
    global boot_run

    f0 = beta_tru
    est = lambda data: lr_est(data, f0)
    priv_est = lambda data, eps: priv_lr_est_approx(data, eps, f0)

    cov_dict_lst = [dict.fromkeys(method_lst, 0) for _ in range(logreg_d_intercept)]
    len_dict_lst = [dict.fromkeys(method_lst, 0) for _ in range(logreg_d_intercept)]
    set_list_dict = dict.fromkeys(method_lst, 0)

    np.random.seed((num_trials * n + trial) % 2**32)
    data_idx = np.random.choice(full_data.shape[0], size=n, replace=False)
    data = full_data[data_idx, :]
    ##### UNSTUDENTIZED #####
    studentize = False
    H_all_dims = []

    for dim in range(logreg_d_intercept):
        search_range_const = search_range_const_lst[dim]
        multiplier = multiplier_lst[dim]
        unstud_low = -search_range_const
        unstud_high = search_range_const
        H_all_dims.append(
            IntervalDiv(
                min=unstud_low,
                max=unstud_high,
                precision=precision_unstud(multiplier, n),
                round_up=round_up_prec,
            )
        )

    if "proper_unstud" in method_lst:
        set_list_dict["proper_unstud"] = fast_abthresh_blb_sens_alldim(
            data,
            H_all_dims,
            b_sens_unstud(n),
            runs_per_boot=runs_per_boot_sens_unstud(n),
            studentize=studentize,
            eps_sigma=None,
            eps_param=eps_param,
            eps_cdf=eps_cdf,
            est=est,
            priv_est=priv_est,
            alpha=alpha,
            set_type=set_type,
            err_correction=err_correction_unstud(n),
        )

    ### Normal approximation ###

    normal_set_lst = [
        Box1D(l=t_dist.ppf(alpha / 2, df=n - 1), u=t_dist.ppf(1 - alpha / 2, df=n - 1))
    ] * logreg_d_intercept

    if "normal_proper_blb" in method_lst:
        priv_noise_blb = priv_noise_n_std_est(data, eps_cdf, f0)
        est_std_for_ci = priv_noise_blb.copy() * np.sqrt(n)

        set_list_dict["normal_proper_blb"] = [
            Box1D(
                l=est_std_for_ci[i] * normal_set_lst[i].l,
                u=est_std_for_ci[i] * normal_set_lst[i].u,
            )
            for i in range(logreg_d_intercept)
        ]

    #### Non private bootstrap estimate unstudentized ###
    if ("boot_unstud" in method_lst) and boot_run:
        set_list_dict["boot_unstud"] = bootstrap_quantile_ci_alldim(
            data,
            n_samples=n,
            n_runs=n_boot(n),
            studentize=False,
            eps_sigma=None,
            eps_param=None,
            alpha=alpha,
            est=est,
        )
    elif "boot_unstud" in method_lst:
        set_list_dict["boot_unstud"] = [Box1D(l=0, u=0)] * logreg_d_intercept
    #### Non private bootstrap for private estimator unstudentized ###
    if ("boot_unstud_priv" in method_lst) and boot_run:
        set_list_dict["boot_unstud_priv"] = bootstrap_quantile_ci_alldim(
            data,
            n_samples=n,
            n_runs=n_boot(n),
            studentize=False,
            eps_sigma=None,
            eps_param=eps_param,
            alpha=alpha,
            est=est,
            priv_est=priv_est,
        )
    elif "boot_unstud_priv" in method_lst:
        set_list_dict["boot_unstud_priv"] = [Box1D(l=0, u=0)] * logreg_d_intercept
    #####################################################################################################

    cov_lst = [[] for _ in range(logreg_d_intercept)]
    len_lst = [[] for _ in range(logreg_d_intercept)]

    ### Update dictionaries ###

    for resamp in range(num_resamp):
        np.random.seed((num_resamp * (num_trials * n + trial) + resamp) % 2**32)

        data_idx = np.random.choice(full_data.shape[0], size=n, replace=False)
        data = full_data[data_idx, :]
        tru_param = beta_tru

        est_nonpriv = est(data)
        est_priv = priv_lr_est_approx(data, eps_param, f0, est_nonpriv.copy())

        for dim_val in range(logreg_d_intercept):
            for key in method_lst:
                est_for_ci = est_priv.copy()

                cov_dict_lst[dim_val][key] = in_set(
                    tru_param[dim_val],
                    est_for_ci[dim_val] + set_list_dict[key][dim_val].l / np.sqrt(n),
                    est_for_ci[dim_val] + set_list_dict[key][dim_val].u / np.sqrt(n),
                )

                len_dict_lst[dim_val][key] = (
                    set_list_dict[key][dim_val].u - set_list_dict[key][dim_val].l
                ) / np.sqrt(n)

            param_lst = [
                n,
                trial,
                resamp,
                eps_total,
                eps_param,
            ]
            cov_lst[dim_val].append(param_lst.copy())
            len_lst[dim_val].append(param_lst.copy())

            for key in method_lst:
                cov_lst[dim_val][-1].append(cov_dict_lst[dim_val][key])
                len_lst[dim_val][-1].append(len_dict_lst[dim_val][key])

    return cov_lst, len_lst


for n in n_lst:
    print(MAX_ACTUAL_RUNS)
    start_n = time.time()
    cov_df = pd.DataFrame(columns=columns)
    len_df = pd.DataFrame(columns=columns)
    cov_lst = [[] for _ in range(logreg_d_intercept)]
    len_lst = [[] for _ in range(logreg_d_intercept)]

    print(n_boot(n))
    if n_boot(n) > MAX_ACTUAL_RUNS:
        print("MAX RUNs LIMIT EXCEEDED")

    if n == 5000:
        num_proc = 28
    elif n == 10000:
        num_proc = 17
    else:
        num_proc = 80

    parallel_rounds = int(np.ceil(num_trials / num_proc))

    tru_trials = parallel_rounds * num_proc

    for par_round in range(parallel_rounds):
        start_trial = time.time()

        pool = mp.Pool(num_proc)
        out_lst = pool.starmap(
            parallel_trials,
            [(n, par_round * num_proc + proc) for proc in range(num_proc)],
        )

        for ot in out_lst:
            for dim_val in range(logreg_d_intercept):
                cov_lst[dim_val].extend(ot[0][dim_val])
                len_lst[dim_val].extend(ot[1][dim_val])

        print(f"{par_round} parallel rounds done in time {time.time() - start_trial}")

    for dim_val in range(logreg_d_intercept):
        cov_df = pd.DataFrame(cov_lst[dim_val], columns=columns)
        len_df = pd.DataFrame(len_lst[dim_val], columns=columns)
        cov_df.attrs["attr_dict"] = attr_dict

        save_dir_dim = save_dir + f"dim{dim_val}/"
        os.system("mkdir -p " + save_dir_dim)
        cov_file_name = save_dir_dim + dist + "_" + str(n) + "_mean_cov.pickle"
        len_file_name = save_dir_dim + dist + "_" + str(n) + "_mean_len.pickle"
        cov_df.to_pickle(cov_file_name)
        len_df.to_pickle(len_file_name)

    print(f"{n} data size done in time {time.time() - start_n}")
