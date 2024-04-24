import argparse

from dp_inference.common_dist_framework import *
from dp_inference.estimators import *
from dp_inference.priv_mech import *


def get_dist():
    parser = argparse.ArgumentParser(description="Collecting the input")
    parser.add_argument(
        "--dist",
        type=str,
        default="truncnorm_assym",
        help="distribution",
        choices=[
            "truncnorm_sym",
            "truncnorm_assym",
            "truncnorm_assym2",
            "truncnorm_assym3",
            "triang_sym",
            "triang_assym",
            "uniform",
        ],
    )
    # Add a boolean argument using the BooleanOptionalAction
    parser.add_argument(
        "--round_up_prec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable a flag",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mean",
        help="Problem to solve",
        choices=["mean", "median", "logreg_synth", "logreg_adult"],
    )
    args = parser.parse_args()

    dist = args.dist
    round_up_prec = args.round_up_prec
    task = args.task
    input_dict = {"dist": dist, "round_up_prec": round_up_prec, "task": task}
    print(input_dict)
    method_lst = [
        "normal_proper_blb",
        # "improper_unstud",
        "proper_unstud",
        "boot_unstud_priv",
    ]
    if task == "mean" or task == "median":
        if dist == "normal":
            P = Gauss1D(mu=0.0, sigma=2.0)
        elif dist == "truncnorm_sym":
            P = TruncGauss1D(trunc_left=-1, trunc_right=1, sigma=5)
        elif dist == "truncnorm_assym":
            P = TruncGauss1D(trunc_left=-3, trunc_right=2, sigma=2)
        elif dist == "truncnorm_assym2":
            P = TruncGauss1D(trunc_left=-4, trunc_right=1, sigma=2)
        elif dist == "truncnorm_assym3":
            P = TruncGauss1D(trunc_left=0, trunc_right=5, sigma=2)
        elif dist == "triang_sym":
            P = Triang1D(c=0.5, left=-5, right=5)
        elif dist == "triang_assym":
            P = Triang1D(c=0.2, left=-5, right=5)

        if task == "mean":
            method_lst += ["normal_proper_adhoc"]

        return P, input_dict, method_lst


def get_estimators(task, P):
    if task == "mean":
        tru_param = P.get_mean()
        tru_std = P.get_std()
        trunc_low, trunc_high = P.get_bounds()
        est = mean1D
        priv_est = lambda X, eps: priv_mean1D(X, trunc_low, trunc_high, eps)
    elif task == "median":
        tru_param = P.get_median()
        tru_std = P.get_std()
        trunc_low, trunc_high = P.get_bounds()
        est = median
        priv_est = lambda X, eps: priv_med_gumbel(
            X.reshape(
                -1,
            ),
            eps=eps,
            R_low=trunc_low,
            R=trunc_high,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return tru_param, tru_std, est, priv_est
