from uncertainties import unumpy as unp
import scipy.optimize
import numdifftools
import numpy as np
import pandas as pd


import scipy.stats
import scipy.special




def extract_bs_spectrum(fit_results_channel, n_w_bins, spectrum_id):
    return unp.uarray(
        [fit_results_channel[i_bin].x[spectrum_id] for i_bin in range(n_w_bins - 1)],
        [fit_results_channel[i_bin].covariance[spectrum_id, spectrum_id]**0.5 for i_bin in range(n_w_bins - 1)]
    )


def determine_migration_martrix(df, bin_edges, col_mc, col_reco, weight_column="__weight_overall__"):
    df.loc[:, f"bins_{col_mc}"] = pd.cut(df[col_mc], bins=bin_edges, labels=range(len(bin_edges)-1))
    df.loc[:, f"bins_{col_reco}"] = pd.cut(df[col_reco], bins=bin_edges, labels=range(len(bin_edges)-1))
    
    return np.array([
        np.array([
            sum(df.query(f"bins_{col_reco} == {i_reco_bin} and bins_{col_mc} == {i_mc_bin}")[weight_column])
            for i_reco_bin in range(len(bin_edges)-1)
        ]) / sum(df.query(f"bins_{col_mc} == {i_mc_bin}")[weight_column])
        for i_mc_bin in range(len(bin_edges)-1)
    ]).transpose()

