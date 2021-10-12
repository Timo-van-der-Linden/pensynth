#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application to Lalonde (1986) dataset

Created on Sun Nov  1 10:51:38 2020

@author: jeremylhour
"""
import numpy as np
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List
from pensynth.pensynthpy import (
    in_hull,
    incremental_pure_synth,
    pensynth_weights
)


def print_between_bars(string: str) -> None:
    print("=" * 80 + "\n" + string + "\n" + "=" * 80)


def print_preamble() -> None:
    now = datetime.now().strftime("%d, %b %Y, %H:%M:%S")
    msg = (
        f"This is a script to compute the pure synthetic control solution for"
        f"Lalonde (1986) data.\nLaunched on {now}"
    )
    print(msg)


def read_data(data_path: str) -> Tuple[np.ndarray]:
    """
    Loading Lalonde's dataset rescaled as in the paper and unscaled for
    statistics.
    """
    x1_full = np.loadtxt(data_path + "x1.txt", skiprows=1)
    y1_full = np.loadtxt(data_path + "y1.txt", skiprows=1)
    x0_full = np.loadtxt(data_path + "x0.txt", skiprows=1)
    y0_full = np.loadtxt(data_path + "y0.txt", skiprows=1)
    x0_unscaled_full = np.loadtxt(data_path + "x0_unscaled.txt", skiprows=1)
    return x1_full, y1_full, x0_full, y0_full, x0_unscaled_full


def construct_dataframe(
    x0_full: np.ndarray,
    y0_full: np.ndarray,
    x0_unscaled_full: np.ndarray,
    x_names: List[str]
) -> pd.DataFrame:
    """
    Consolidate duplicates in x0
    """
    df = pd.DataFrame(x0_full)
    df.columns = [item + "_rescaled" for item in x_names]
    df["outcome"] = y0_full
    for i in range(x0_unscaled_full.shape[1]):
        df[x_names[i]] = x0_unscaled_full[:, i]
    return df


def remove_duplicates(
    df: pd.DataFrame,
    x_names: List[str]
) -> Tuple[np.ndarray]:
    """
    Consolidate dataset for untreated
    """
    df_consolidated = df.groupby(x_names, as_index=False).mean()
    x0 = df_consolidated[[i + "_rescaled" for i in x_names]].to_numpy()
    x0_unscaled = df_consolidated[x_names].to_numpy()
    y0 = df_consolidated["outcome"].to_numpy()
    return x0, x0_unscaled, y0


def compute_statistics(
    all_w: np.ndarray,
    x0_unscaled: np.ndarray,
    y0: np.ndarray,
    y1_full: np.ndarray
) -> Tuple[np.ndarray]:
    """
    COMPUTE THE NECESSARY STATISTICS
    """
    y0_hat = all_w @ y0
    balance_check = (all_w @ x0_unscaled).mean(axis=0)

    print("ATT: {:.3f}".format((y1_full - y0_hat).mean(axis=0)))

    for b, value in enumerate(balance_check):
        print(x_names[b] + ": {:.3f}".format(value))

    sparsity_index = (all_w > 0).sum(axis=1)
    print("Min sparsity: {:.0f}".format(sparsity_index.min()))
    print("Median sparsity: {:.0f}".format(np.median(sparsity_index)))
    print("Max sparsity: {:.0f}".format(sparsity_index.max()))

    activ_index = (all_w > 0).sum(axis=0) > 0
    print("Active untreated units: {:.0f}".format(activ_index.sum()))
    return y0_hat, balance_check, sparsity_index


def run_optimizer(x0, x1_full) -> np.ndarray:
    """
    We proceed in 3 steps:
    - if some untreated are the same as the treated, assign uniform weights to
      these untreated.
    - if the treated is inside the convex hull defined by the untreated, run
      the incremental algo.
    - if the treated is not inside the convex hull defined by the untreated,
      run the standard synthetic control.
    """
    all_w = np.zeros((len(x1_full), len(x0)))
    start_time = time.time()
    with tqdm(total=(len(x1_full))) as prog:
        for i, x in enumerate(x1_full):
            # True if untreated is same as treated
            sameAsUntreated = np.all(x0 == x, axis=1)
            if any(sameAsUntreated):
                untreatedId = np.where(sameAsUntreated)
                all_w[i, untreatedId] = 1 / len(untreatedId)
            else:
                inHullFlag = in_hull(x=x, points=x0)
                if inHullFlag:
                    x0_tilde, antiranks = incremental_pure_synth(
                        X1=x,
                        X0=x0
                    )
                    all_w[i, antiranks] = pensynth_weights(
                        X0=x0_tilde,
                        X1=x,
                        pen=0
                    )
                else:
                    all_w[i, ] = pensynth_weights(X0=x0, X1=x, pen=1e-6)
            prog.update(1)
    time_elapsed = time.time() - start_time
    print(f"Time elapsed : {time_elapsed:.2f} seconds ---")
    return all_w


def save_weights_as_parquet(all_w: np.ndarray, x0: np.ndarray) -> None:
    """
    SAVING WEIGHTS AS PARQUET FILE
    """
    df = pd.DataFrame(all_w)
    df.columns = ["Unit_" + str(i + 1) for i in range(len(x0))]
    df.to_parquet("Lalonde_solution.parquet", engine="pyarrow")


def statistics_sanity_check(sparsity_index: np.ndarray) -> np.ndarray:
    """
    SANITY CHECK ON SPARSITY
    """
    high_sparsity = np.where(sparsity_index > 11)[0]
    print(f"{len(high_sparsity)} treated units have sparsity larger than p+1.")
    print(high_sparsity)
    return high_sparsity


def write_statistics_to_file(
    y1_full: np.ndarray,
    y0_hat: np.ndarray,
    balance_check: np.ndarray,
    sparsity_index: np.ndarray,
    high_sparsity: np.ndarray
) -> None:
    """
    DUMPING STATS TO FILE
    """
    with open("statistics.txt", "w") as f:
        f.write("ATT: {:.3f}\n".format((y1_full - y0_hat).mean(axis=0)))
        for b, value in enumerate(balance_check):
            f.write(x_names[b] + ": {:.3f}\n".format(value))
        f.write("Min sparsity: {:.0f}\n".format(sparsity_index.min()))
        f.write("Median sparsity: {:.0f}\n".format(np.median(sparsity_index)))
        f.write("Max sparsity: {:.0f}\n".format(sparsity_index.max()))
        f.write(f"{len(high_sparsity)} treated units have sparsity larger than p+1.")


if __name__ == "__main__":
    print_between_bars("DATA MANAGEMENT")
    x_names = [
        "age",
        "education",
        "married",
        "black",
        "hispanic",
        "re74",
        "re75",
        "nodegree",
        "NoIncome74",
        "NoIncome75"
    ]
    x1_full, y1_full, x0_full, y0_full, x0_unscaled_full = read_data("data/")
    df = construct_dataframe(x0_full, y0_full, x0_unscaled_full, x_names)
    x0, x0_unscaled, y0 = remove_duplicates(df, x_names)
    print_between_bars("COMPUTING PURE SYNTHETIC CONTROL FOR EACH TREATED")
    all_w = run_optimizer(x0, x1_full)
    print_between_bars("COMPUTING STATISTICS AND SAVING RESULTS")
    y0_hat, balance_check, sparsity_index = compute_statistics(
        all_w,
        x0_unscaled,
        y0,
        y1_full
    )
    save_weights_as_parquet(all_w, x0)
    high_sparsity = statistics_sanity_check(sparsity_index)
    write_statistics_to_file(
        y1_full,
        y0_hat,
        balance_check,
        sparsity_index,
        high_sparsity
    )
