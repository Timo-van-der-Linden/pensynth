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
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List, Optional
from pathlib import Path
from box import Box
from pensynth.pensynthpy import (
    in_hull,
    incremental_pure_synth,
    pensynth_weights
)


@dataclass
class LalondeData():
    path: Path
    x0: Optional[pd.DataFrame] = None
    x1: Optional[pd.DataFrame] = None
    y0: Optional[pd.DataFrame] = None
    y1: Optional[pd.DataFrame] = None
    x0_unscaled: Optional[pd.DataFrame] = None
    scaling: Optional[pd.Series] = None
    data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.read_raw_data()
        self.determine_scaling()
        self.combine_data()

    def read_raw_data(self) -> None:
        """
        Loading Lalonde's dataset rescaled as in the paper and unscaled for
        statistics.
        """
        names = dict(
            x0="X0",
            x1="X1",
            y0="Y0",
            y1="Y1",
            x0_unscaled="X0_unscaled"
        )
        for name, file in names.items():
            data = pd.read_csv(self.path / (file + ".txt"), sep=" ")
            self.__setattr__(name, data)

    def determine_scaling(self) -> None:
        scaling = (self.x0_unscaled / self.x0).mean() ** -1
        scaling["treatment"] = 1
        scaling["outcome"] = 1
        self.scaling = scaling

    def _average_duplicates(
        self,
        data: pd.DataFrame,
        columns_to_average: List[str]
    ) -> pd.DataFrame:
        may_be_duplicate = list(data.columns.difference(columns_to_average))
        unique_index = data[~data[may_be_duplicate].duplicated()].index
        result = data.groupby(may_be_duplicate)[data.columns].mean()
        result.index = unique_index
        return result

    def _get_treated(self) -> pd.DataFrame:
        result = pd.concat([self.x1, self.y1], axis=1)
        result["treatment"] = 1
        result.rename(columns={"x": "outcome"}, inplace=True)
        return result

    def _get_untreated(self) -> pd.DataFrame:
        result = pd.concat([self.x0, self.y0], axis=1)
        result["treatment"] = 0
        result.rename(columns={"x": "outcome"}, inplace=True)
        return result

    def combine_data(self) -> None:
        treated = self._get_treated()
        untreated = self._get_untreated()
        untreated.index = untreated.index + treated.shape[0]
        untreated = self._average_duplicates(untreated, ["outcome"])
        result = pd.concat([treated, untreated]) / self.scaling
        self.data = result[self.scaling.index]
        


@dataclass
class MyData():
    data: pd.DataFrame
    outcome_var: str
    treatment_var: str
    scaling: Optional[pd.Series]

    @property
    def scaled(self) -> pd.DataFrame:
        if self.scaling is None:
            scaled = (self.data - self.data.mean()) / self.data.std()
        else:
            scaled = self.data * self.scaling
        return scaled

    @property
    def outcome(self) -> pd.Series:
        return self.data[self.outcome_var]

    @property
    def treatment(self) -> pd.Series:
        return self.data[self.treatment_var]

    @property
    def x_names(self) -> list:
        to_drop = [self.outcome_var, self.treatment_var]
        return self.data.drop(columns=to_drop).columns.to_list()

    @property
    def x_control(self) -> pd.DataFrame:
        return self.scaled.loc[self.treatment == 0, self.x_names].values

    @property
    def x_treatment(self) -> pd.DataFrame:
        return self.scaled.loc[self.treatment == 1, self.x_names].values

    @property
    def y_control(self) -> pd.Series:
        return self.outcome[self.treatment == 0].values

    @property
    def y_treatment(self) -> pd.Series:
        return self.outcome[self.treatment == 1].values


def print_between_bars(string: str) -> None:
    print("=" * 80 + "\n" + string + "\n" + "=" * 80)


def print_preamble() -> None:
    now = datetime.now().strftime("%d, %b %Y, %H:%M:%S")
    msg = (
        f"This is a script to compute the pure synthetic control solution for"
        f"Lalonde (1986) data.\nLaunched on {now}"
    )
    print(msg)


def construct_dataframe(data: Box) -> pd.DataFrame:
    """
    Consolidate duplicates in x_untreated
    """
    x_names = data.x_untreated_full.columns.to_list()
    df = data.x_untreated_full.copy()
    df.columns = [i + "_rescaled" for i in data.x_untreated_full.columns]
    df["outcome"] = data.y_untreated_full
    for i in range(data.x_untreated_unscaled_full.shape[1]):
        df[x_names[i]] = data.x_untreated_unscaled_full.iloc[:, i]
    return df


def remove_duplicates(data: Box) -> Tuple[np.ndarray]:
    """
    Consolidate dataset for untreated
    """
    x_names = data.x_untreated_full.columns.to_list()
    df_consolidated = data.df.groupby(x_names, as_index=False).mean()
    x_untreated = df_consolidated[[i + "_rescaled" for i in x_names]]
    x_untreated_unscaled = df_consolidated[x_names]
    y_untreated = df_consolidated["outcome"]
    return x_untreated, x_untreated_unscaled, y_untreated


def compute_statistics(
    weights: np.ndarray,
    x_untreated_unscaled: np.ndarray,
    y_untreated: np.ndarray,
    y_treated_full: np.ndarray,
    x_names
) -> Tuple[np.ndarray]:
    """
    COMPUTE THE NECESSARY STATISTICS
    """
    y_synthetic_control = weights @ y_untreated
    balance_check = (weights @ x_untreated_unscaled).mean(axis=0)

    print("ATT: {:.3f}".format((y_treated_full - y_synthetic_control).mean(axis=0)))

    for b, value in enumerate(balance_check):
        print(x_names[b] + ": {:.3f}".format(value))

    sparsity_index = (weights > 0).sum(axis=1)
    print("Min sparsity: {:.0f}".format(sparsity_index.min()))
    print("Median sparsity: {:.0f}".format(np.median(sparsity_index)))
    print("Max sparsity: {:.0f}".format(sparsity_index.max()))

    activ_index = (weights > 0).sum(axis=0) > 0
    print("Active untreated units: {:.0f}".format(activ_index.sum()))
    return y_synthetic_control, balance_check, sparsity_index


def run_optimizer(x_untreated, x_treated_full) -> np.ndarray:
    """
    We proceed in 3 steps:
    - if some untreated are the same as the treated, assign uniform weights to
      these untreated.
    - if the treated is inside the convex hull defined by the untreated, run
      the incremental algo.
    - if the treated is not inside the convex hull defined by the untreated,
      run the standard synthetic control.
    """
    weights = np.zeros((len(x_treated_full), len(x_untreated)))
    start_time = time.time()
    with tqdm(total=(len(x_treated_full))) as prog:
        for i, x in enumerate(x_treated_full):
            # True if untreated is same as treated
            sameAsUntreated = np.all(x_untreated == x, axis=1)
            if any(sameAsUntreated):
                untreatedId = np.where(sameAsUntreated)
                weights[i, untreatedId] = 1 / len(untreatedId)
            else:
                inHullFlag = in_hull(x=x, points=x_untreated)
                if inHullFlag:
                    x_untreated_tilde, antiranks = incremental_pure_synth(
                        X1=x,
                        X0=x_untreated
                    )
                    weights[i, antiranks] = pensynth_weights(
                        X0=x_untreated_tilde,
                        X1=x,
                        pen=0
                    )
                else:
                    weights[i, ] = pensynth_weights(
                        X0=x_untreated,
                        X1=x, pen=1e-6
                    )
            prog.update(1)
    time_elapsed = time.time() - start_time
    print(f"Time elapsed : {time_elapsed:.2f} seconds ---")
    return weights


def save_weights_as_parquet(weights: np.ndarray, x_untreated: np.ndarray) -> None:
    """
    SAVING WEIGHTS AS PARQUET FILE
    """
    df = pd.DataFrame(weights)
    df.columns = ["Unit_" + str(i + 1) for i in range(len(x_untreated))]
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
    y_treated_full: np.ndarray,
    y_synthetic_control: np.ndarray,
    balance_check: np.ndarray,
    sparsity_index: np.ndarray,
    high_sparsity: np.ndarray,
    x_names: List
) -> None:
    """
    DUMPING STATS TO FILE
    """
    with open("statistics.txt", "w") as f:
        f.write("ATT: {:.3f}\n".format((y_treated_full - y_synthetic_control).mean(axis=0)))
        for b, value in enumerate(balance_check):
            f.write(x_names[b] + ": {:.3f}\n".format(value))
        f.write("Min sparsity: {:.0f}\n".format(sparsity_index.min()))
        f.write("Median sparsity: {:.0f}\n".format(np.median(sparsity_index)))
        f.write("Max sparsity: {:.0f}\n".format(sparsity_index.max()))
        f.write(f"{len(high_sparsity)} treated units have sparsity larger than p+1.")


if __name__ == "__main__":
    print_between_bars("DATA MANAGEMENT")
    lalonde = LalondeData(Path("./data"))
    data = MyData(
        data=lalonde.data,
        outcome_var="outcome",
        treatment_var="treatment",
        scaling=lalonde.scaling
    )
    print_between_bars("COMPUTING PURE SYNTHETIC CONTROL FOR EACH TREATED")
    weights = run_optimizer(data.x_control, data.x_treatment)
    # weights = pd.read_csv(path / "weights.csv").values
    print_between_bars("COMPUTING STATISTICS AND SAVING RESULTS")
    y0_hat, balance_check, sparsity_index = compute_statistics(
        weights,
        data.data.loc[data.treatment == 0, data.x_names],
        data.y_control,
        data.y_treatment,
        data.x_names
    )
    save_weights_as_parquet(weights, data.x_control)
    high_sparsity = statistics_sanity_check(sparsity_index)
    write_statistics_to_file(
        data.y_treatment,
        y0_hat,
        balance_check,
        sparsity_index,
        high_sparsity,
        data.x_names
    )
