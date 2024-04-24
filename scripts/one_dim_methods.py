import os

import numpy as np
from scipy.stats import t as t_dist

import pandas as pd

from dp_inference.common_dist_framework import IntervalDiv, Box1D

from dp_inference.inference_methods import (
    fast_abthresh_blb_sens,
    bootstrap_quantile_ci,
)
from dp_inference.estimators import (
    priv_blb_std_est1D,
    std_mean1D,
    priv_std_mean1D,
)
from dp_inference.priv_mech import in_set
import dp_inference.utils as utils

import time

import multiprocessing as mp
from datetime import date

########### DATA PARAMETERS ###########
P, input_dict, method_lst = utils.get_dist()
task = input_dict["task"]
trunc_low, trunc_high = P.get_bounds()
################ Parameters ##########
n_lst = [300, 500, 1000, 1500, 2000]


const_std_up = 50
const_inv_sens_unstud = 10

min_boots = 100
max_boots = 10000

alpha = 0.05
mc_exp = 1.5
gran_exp = 0.5

num_trials = 1000
num_resamp = 100
search_range_const = 5

multiplier = 1
set_type = "symm"  # "eqtail" or "symm"

####### PRIVACY PARAMETERS ########
eps_total_lst = [10, 8, 5]
eps_param_frac_lst = [0.5]

################### ESTIMATORS ####################
true_param, true_std, est, priv_est = utils.get_estimators(task, P)

priv_noise_est = lambda X, eps: priv_blb_std_est1D(
    X,
    b=b_sens_unstud(len(X)),
    est=lambda X: priv_est(X, eps_param),
    runs_per_boot=runs_per_boot_sens_unstud(len(X)),
    eps_sigma=eps,
    R_upp_var=(const_std_up * true_std / np.sqrt(len(X))) ** 2,
    rho_granularity=1 / len(X) ** (gran_exp + 1),
)

if task == "mean":
    trunc_low, trunc_high = P.get_bounds()
    std_est = std_mean1D
    priv_std_est = lambda X, eps: priv_std_mean1D(X, trunc_low, trunc_high, eps)

########### ALG PARAMETERS ###########

boot_run = True
oracle_min_length_unstud = 0
err_correction_unstud = lambda n: 0  # np.log(n) / n  # 1 / n  # np.log(n)/n**0.5
precision_unstud = lambda multiplier, n: multiplier / (n**0.5)

################# OTHER BOOTSTRAP HYPERPARAMETERS LIST ################

if set_type == "eqtail":
    const_inv_sens_unstud *= 2

s_sens_unstud = lambda n: np.ceil(const_inv_sens_unstud * np.log(n) / eps_cdf_unstud)

b_sens_unstud = lambda n: int(n / s_sens_unstud(n))
runs_per_boot_sens_unstud = lambda n: int(
    max(
        min(
            n ** (mc_exp - 1) * b_sens_unstud(n) / (np.log(n)),
            max_boots,
        ),
        min_boots,
    )
)

n_boot = lambda n: int(max(min(n**mc_exp / (np.log(n)), max_boots), min_boots))


### Versioning folder name ###
dir_prefix = f"results/Final_{task}_est_{set_type}_sided_cov_{1 - alpha}_and_len/"
os.system("mkdir -p " + dir_prefix)

day = date.today().day
month = date.today().month

###################### Columns for saving results ############################
columns = [
    "n",
    "trial",
    "resamp",
    "eps_total",
    "eps_param_frac",
]

########### Attribute Dict #############
attr_dict = input_dict | {
    "set_type": set_type,
    "alpha": alpha,
    "mc_exp": mc_exp,
    "gran_exp": gran_exp,
    "const_std_up": const_std_up,
    "const_inv_sens_unstud": const_inv_sens_unstud,
}

columns.extend(method_lst)


def parallel_trials(n, trial, P):
    np.random.seed((num_trials * n + trial) % 2**32)
    data = P.sample(n)

    cov_dict = dict.fromkeys(method_lst, 0)
    len_dict = dict.fromkeys(method_lst, 0)
    set_dict = dict.fromkeys(method_lst, 0)
    # cov_arr_dict = {key: [] for key in sets_keys}
    # len_arr_dict = {key: [] for key in sets_keys}

    ##### STUDENTIZED #####
    studentize = True

    # Normal Sets
    normal_set = Box1D(
        l=t_dist.ppf(alpha / 2, df=n - 1), u=t_dist.ppf(1 - alpha / 2, df=n - 1)
    )
    # correcting for added noise to get new variance
    # sqrt(n) is important because dividing by sqrt(n)
    # outside later
    noise_std = np.sqrt(2) * (trunc_high - trunc_low) / (np.sqrt(n) * eps_param)

    if "normal_proper_blb" in method_lst:
        priv_noise_blb = priv_noise_est(data, eps_sigma_norm)
        std_val = priv_noise_blb * np.sqrt(n)
        set_dict["normal_proper_blb"] = Box1D(
            l=std_val * normal_set.l, u=std_val * normal_set.u
        )

    if "normal_proper_adhoc" in method_lst:
        priv_stddev_norm = priv_std_est(data, eps_sigma_norm)
        std_val = np.hypot(priv_stddev_norm, noise_std)
        set_dict["normal_proper_adhoc"] = Box1D(
            l=std_val * normal_set.l, u=std_val * normal_set.u
        )

    ##### UNSTUDENTIZED #####
    studentize = False
    unstud_low = -search_range_const * true_std
    unstud_high = search_range_const * true_std
    H_unstud = IntervalDiv(
        min=unstud_low,
        max=unstud_high,
        precision=precision_unstud(multiplier, n),
        round_up=input_dict["round_up_prec"],
    )
    if "proper_unstud" in method_lst:
        set_dict["proper_unstud"] = fast_abthresh_blb_sens(
            data,
            H_unstud,
            b_sens_unstud(n),
            runs_per_boot_sens_unstud(n),
            studentize=studentize,
            eps_sigma=None,
            eps_param=eps_param,
            eps_cdf=eps_cdf_unstud,
            est=est,
            priv_est=priv_est,
            oracle_min_length=oracle_min_length_unstud,
            alpha=alpha,
            set_type=set_type,
            err_correction=err_correction_unstud(n),
        )
    #### Non private bootstrap estimate unstudentized ###
    if ("boot_unstud" in method_lst) and boot_run:
        set_dict["boot_unstud"] = bootstrap_quantile_ci(
            data,
            n_samples=n,
            n_runs=n_boot(n),
            studentize=False,
            eps_sigma=None,
            eps_param=None,
            est=est,
            alpha=alpha,
            set_type=set_type,
        )
    elif "boot_unstud" in method_lst:
        set_dict["boot_unstud"] = Box1D(l=0, u=0)

    if ("boot_unstud_priv" in method_lst) and boot_run:
        set_dict["boot_unstud_priv"] = bootstrap_quantile_ci(
            data,
            n_samples=n,
            n_runs=n_boot(n),
            studentize=False,
            eps_sigma=None,
            eps_param=eps_param,
            est=est,
            priv_est=priv_est,
            alpha=alpha,
            set_type=set_type,
        )
    elif "boot_unstud_priv" in method_lst:
        set_dict["boot_unstud_priv"] = Box1D(l=0, u=0)

    cov_lst, len_lst = [], []
    ### Update dictionaries ###

    for resamp in range(num_resamp):
        np.random.seed((num_resamp * (num_trials * n + trial) + resamp) % 2**32)
        data = P.sample(n)

        priv_est_val = priv_est(data, eps_param)

        est_val = priv_est_val
        for key in method_lst:
            cov_dict[key] = in_set(
                true_param,
                est_val + set_dict[key].l / np.sqrt(n),
                est_val + set_dict[key].u / np.sqrt(n),
            )

            len_dict[key] = np.abs(set_dict[key].u - set_dict[key].l) / np.sqrt(n)

        # param list to save to dataframe
        param_lst = [
            n,
            trial,
            resamp,
            eps_total,
            eps_param_frac,
        ]

        cov_lst.append(param_lst.copy())
        len_lst.append(param_lst.copy())

        for key in method_lst:
            cov_lst[-1].append(cov_dict[key])
            len_lst[-1].append(len_dict[key])

    return cov_lst, len_lst


count = 0

for eps_total in eps_total_lst:
    start_eps_total = time.time()
    for eps_param_frac in eps_param_frac_lst:
        start_eps_param = time.time()
        eps_param = eps_param_frac * eps_total
        eps_cdf_unstud = eps_total - eps_param
        eps_cdf_stud = (eps_total - eps_param) / 2
        eps_sigma = (eps_total - eps_param) / 2
        eps_sigma_norm = eps_cdf_unstud

        # Versioning files
        version = 0

        file_list = [
            file for file in os.listdir(dir_prefix) if f"m{month}d{day}" in file
        ]
        version += len(file_list)

        save_dir = f"{dir_prefix}m{month}d{day}v{version}/"

        os.system("mkdir -p " + save_dir)

        if count > 0:
            boot_run = False

        for n in n_lst:
            start_n = time.time()
            cov_df = pd.DataFrame(columns=columns)
            len_df = pd.DataFrame(columns=columns)
            cov_lst = []
            len_lst = []

            num_proc = 80

            parallel_rounds = int(np.ceil(num_trials / num_proc))

            tru_trials = parallel_rounds * num_proc

            # out = parallel_trials(n, 1, P)
            print("Starting parallel trials")
            for par_round in range(parallel_rounds):
                start_trial = time.time()

                pool = mp.Pool(num_proc)
                out_lst = pool.starmap(
                    parallel_trials,
                    [(n, par_round * num_proc + proc, P) for proc in range(num_proc)],
                )

                for ot in out_lst:
                    cov_lst.extend(ot[0])
                    len_lst.extend(ot[1])

                print(
                    f"{par_round} parallel rounds done in time {time.time() - start_trial}"
                )

            cov_df = pd.DataFrame(cov_lst, columns=columns)
            len_df = pd.DataFrame(len_lst, columns=columns)
            cov_df.attrs["attr_dict"] = attr_dict

            cov_file_name = save_dir + str(n) + f"_cov.pickle"
            len_file_name = save_dir + str(n) + f"_len.pickle"
            print(f"Computation done in time {time.time() - start_n}, saving now")
            cov_df.to_pickle(cov_file_name)
            len_df.to_pickle(len_file_name)

            print(f"{n} data size done in time {time.time() - start_n}")

        print(
            f"{eps_param} parameter epsilon with {eps_total} total epsilon \
              done in time {time.time() - start_eps_param}"
        )

    print(
        f"{eps_total} total epsilon \
            done in time {time.time() - start_eps_total}"
    )
