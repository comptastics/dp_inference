import os
import numpy as np
import pandas as pd

from dp_inference.gvdp_blb_coinpress import *

import argparse
import time

import multiprocessing as mp
from datetime import date

alpha = 0.05
s_exp = 0.5

eps_total_lst = [10, 8, 5]
failure_prob = 0.01
const_s_gvdp = 5

mc_exp = 1.5

num_trials = 10000
num_resamp = 1
MAX_ACTUAL_RUNS = 1000000

parser = argparse.ArgumentParser(description="Collecting the input")
parser.add_argument(
    "--dist",
    type=str,
    default=0,
    help="distribution",
    choices=[
        "truncnorm_sym",
        "truncnorm_assym",
        "triang_sym",
        "triang_assym",
    ],
)

parser.add_argument(
    "--symm",
    type=int,
    default=1,
)
parser.add_argument(
    "--round",
    type=int,
    default=0,
)

args = parser.parse_args()

dist = args.dist
symm = args.symm
round_up = args.round
print(dist)

if round_up == 0:
    round_up_prec = False
else:
    round_up_prec = True
if symm == 0:
    symm_set = False
else:
    symm_set = True
print(symm_set)


if dist == "normal":
    P = Gauss1D(mu=0.0, sigma=2.0)
elif dist == "truncnorm_sym":
    P = TruncGauss1D(trunc_left=-1, trunc_right=1, sigma=5)
    trunc_low = -5
    trunc_high = 5
elif dist == "truncnorm_assym":
    P = TruncGauss1D(trunc_left=-3, trunc_right=2, sigma=2)
    trunc_low = -6
    trunc_high = 4
elif dist == "triang_sym":
    P = Triang1D(c=0.5, left=-5, right=5)
    trunc_low = -5
    trunc_high = 5
elif dist == "triang_assym":
    P = Triang1D(c=0.2, left=-5, right=5)
    trunc_low = -5
    trunc_high = 5

################### ESTIMATORS ####################
est = median

################ N  LIST ###########################
n_lst = [300, 500, 1000, 1500, 2000]

################# OTHER BOOTSTRAP HYPERPARAMETERS LIST ################
s_gvdp = lambda n: np.ceil(min(const_s_gvdp / eps_total * n**s_exp, n / 5)).astype(int)
b_gvdp = lambda n: int(n / s_gvdp(n))

runs_per_boot_gvdp = lambda n: int(n ** (mc_exp - 1) * b_gvdp(n) / np.log(n))
gvdp_overest_factor = 10

mean_radius_ub = None

### Versioning folder name ###
dir_prefix = f"results/Final_median_est_gvdp_cov_{1 - alpha}_and_len/"
os.system("mkdir -p " + dir_prefix)

day = date.today().day
month = date.today().month

########### Attribute Dict #############
attr_dict = {
    "dist": dist,
    "alpha": alpha,
    "s_exp": s_exp,
    "mc_exp": mc_exp,
    "num_trials": num_trials,
    "num_resamp": num_resamp,
    "MAX_ACTUAL_RUNS": MAX_ACTUAL_RUNS,
    "gvdp_overest_factor": gvdp_overest_factor,
    "mean_radius_ub": mean_radius_ub,
    "const_s_gvdp": const_s_gvdp,
}

### EXPERIMENTS ###
columns = [
    "n",
    "trial",
    "resamp",
    "dist",
    "eps_total",
    "gvdp",
]


def parallel_trials(n, trial, P):
    global precision_stud
    global precision_unstud
    global dist
    global boot_run

    np.random.seed((num_trials * n + trial) % 2**32)
    data = P.sample(n)

    tru_param = P.get_mean()

    mean_arr, cov_arr = blb_means_covs_1D(
        data, b_gvdp(n), runs_per_boot_gvdp(n), estimator=est
    )

    cov_cov_u = gvdp_overest_factor * np.array(np.cov(np.array(cov_arr).T))
    cov_c = np.mean(cov_arr, axis=0)
    cov_r = gvdp_overest_factor * np.mean(cov_arr, axis=0)
    mean_c = np.mean(mean_arr, axis=0)

    # set your own mean_radius upper bound if it is not None
    if mean_radius_ub:
        mean_r = mean_radius_ub
    else:
        mean_r = gvdp_overest_factor * np.max(np.abs(mean_arr))
    t_cov = 5
    t_mean = 5
    (
        rho_mean_budget_prop,
        rho_cov_budget_prop,
        beta_mean_budget_prop,
        beta_cov_budget_prop,
    ) = (0.5, 0.5, 0.5, 0.5)

    #####################################################################################################
    ############## GVDP #########################
    #####################################################################################################
    est_p, priv_var, est_priv_cov, gvdp_set = general_valid_dp(
        data,
        s_gvdp(n),
        runs_per_boot_gvdp(n),
        cov_cov_u=cov_cov_u,
        cov_c=cov_c,
        cov_r=cov_r,
        mean_c=mean_c,
        mean_r=mean_r,
        t_cov=t_cov,
        t_mean=t_mean,
        rho=eps_to_rho(eps_total, 1 / n**1.1),
        rho_mean_budget_prop=rho_mean_budget_prop,
        rho_cov_budget_prop=rho_cov_budget_prop,
        beta=failure_prob,
        beta_mean_budget_prop=beta_mean_budget_prop,
        beta_cov_budget_prop=beta_cov_budget_prop,
        ci_alphas=alpha,
        estimator=est,
    )
    #####################################################################################################
    ############## GVDP #########################
    #####################################################################################################

    cov_lst, len_lst = [], []
    ### Update dictionaries ###

    for resamp in range(num_resamp):
        cov_lst.append(
            [
                n,
                trial,
                resamp,
                dist,
                eps_total,
                in_set(
                    tru_param,
                    gvdp_set[0][0],
                    gvdp_set[0][1],
                ),
            ]
        )
        len_lst.append(
            [
                n,
                trial,
                resamp,
                dist,
                eps_total,
                gvdp_set[0][1] - gvdp_set[0][0],
            ]
        )

    return cov_lst, len_lst


for eps_total in eps_total_lst:
    start_eps_total = time.time()
    # Versioning files
    version = 0

    file_list = [file for file in os.listdir(dir_prefix) if f"m{month}d{day}" in file]
    version += len(file_list)

    save_dir = f"{dir_prefix}m{month}d{day}v{version}/"

    os.system("mkdir -p " + save_dir)

    for n in n_lst:
        print(MAX_ACTUAL_RUNS)
        start_n = time.time()
        cov_df = pd.DataFrame(columns=columns)
        len_df = pd.DataFrame(columns=columns)
        cov_lst = []
        len_lst = []

        num_proc = 80

        parallel_rounds = int(np.ceil(num_trials / num_proc))

        tru_trials = parallel_rounds * num_proc

        start_trial = time.time()
        for par_round in range(parallel_rounds):
            pool = mp.Pool(num_proc)
            out_lst = pool.starmap(
                parallel_trials,
                [(n, par_round * num_proc + proc, P) for proc in range(num_proc)],
            )

            for ot in out_lst:
                cov_lst.extend(ot[0])
                len_lst.extend(ot[1])

            if par_round % 10 == 0:
                print(
                    f"{par_round} parallel rounds done in time {time.time() - start_trial}"
                )
                start_trial = time.time()

        cov_df = pd.DataFrame(cov_lst, columns=columns)
        len_df = pd.DataFrame(len_lst, columns=columns)
        cov_df.attrs["attr_dict"] = attr_dict

        cov_file_name = save_dir + dist + "_" + str(n) + f"_mean_cov.pickle"
        len_file_name = save_dir + dist + "_" + str(n) + f"_mean_len.pickle"
        print(f"Computation done in time {time.time() - start_n}, saving now")
        cov_df.to_pickle(cov_file_name)
        len_df.to_pickle(len_file_name)

        print(f"{n} data size done in time {time.time() - start_n}")

    print(
        f"{eps_total} total epsilon \
            done in time {time.time() - start_eps_total}"
    )
