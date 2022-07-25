import numpy as np


def form_factor_systematics(
    df, fit_variable, fit_bins, fit_range,
    nominal_weight_column_for_systematic, 
    varied_up_weight_columns,
    varied_down_weight_columns,
    weight_column="__weight_overall__"
):
    C = []
    for v_up, v_down in zip(varied_up_weight_columns, varied_down_weight_columns):
        C.append(get_cov_mat_from_up_down_variation(
            df[fit_variable], 
            fit_bins,
            df[weight_column],
            df[nominal_weight_column_for_systematic],
            df[v_up],
            df[v_down]
        ))
    covariance = sum(C)
    systematics = {}
    systematics["events"] = np.histogram(df[fit_variable], bins=fit_bins, weights=df.__weight_overall__)[0]
    systematics["uncertainty"] = covariance.diagonal()**0.5
    systematics["correlation_matrix"] = covariance / np.outer(covariance.diagonal()**0.5, covariance.diagonal()**0.5)
    systematics["covariance_matrix"] = covariance

    return systematics

def get_varied_weights(
    weights: np.array, nominal_weights: np.array, variation_weights: np.array
) -> np.array:
    """Calculates new varied weights by replacing the nominal
    weight with its variation in the given weight. Assumes
    that `weights` is a product of a set of weights including
    `nominal_weights`.
    """
    return np.nan_to_num(weights * (variation_weights / nominal_weights))

def get_varied_histogram(
    data: np.array,
    bins,
    weights: np.array,
    nominal_weights: np.array,
    variation_weights: np.array,
) -> np.array:
    varied_weights = get_varied_weights(weights, nominal_weights, variation_weights)
    varied_hist, _ = np.histogram(data, bins=bins, weights=varied_weights)
    return varied_hist

def get_cov_mat_from_up_down_variation(
    data: np.array,
    bins,
    weights: np.array,
    nominal_weights: np.array,
    up_weights: np.array,
    down_weights: np.array,
) -> np.ndarray:

    up_hist = get_varied_histogram(
        data,
        bins=bins,
        weights=weights,
        nominal_weights=nominal_weights,
        variation_weights=up_weights,
    )
    down_hist = get_varied_histogram(
        data,
        bins=bins,
        weights=weights,
        nominal_weights=nominal_weights,
        variation_weights=down_weights,
    )
    sym_diff = (up_hist - down_hist) / 2.0

    return np.outer(sym_diff, sym_diff)


def transform_covariance_matrix_for_plotting_to_show_only_shape_uncertainty(t, cov):
    N = sum(t)
    e, D = np.linalg.eig(cov)
    Dinv = np.linalg.inv(D)
    
    C = []
    for i, eigenvalue in enumerate(e):
        variation_vector = np.zeros(len(e))
        variation_vector[i] = 1
        var_u = D @ (Dinv @ t + eigenvalue**0.5 * variation_vector)
        var_u = var_u / sum(var_u) * N
        var_d = D @ (Dinv @ t - eigenvalue**0.5 * variation_vector)
        var_d = var_d / sum(var_d) * N
        C.append((abs(t - var_u) + abs(t - var_d))/2)
    
    C = sum(np.array([c**2 for c in C]))
    return cov
    return np.diag(C)


def add_combined_channel(*systematics):
    combined_systematics = {}
    combined_systematics["events"] = sum(s["events"] for s in systematics)
    combined_systematics["uncertainty"] = sum(s["uncertainty"]**2 for s in systematics)**0.5
    covariance = sum(s["covariance_matrix"] for s in systematics)
    combined_systematics["correlation_matrix"] = covariance / np.outer(covariance.diagonal()**0.5, covariance.diagonal()**0.5)
    combined_systematics["covariance_matrix"] = covariance
    return combined_systematics


def variation_systematics(df, fit_variable, fit_bins, fit_range, nominal_weight_column_for_systematic, varied_weight_columns, weight_column="__weight_overall__"):
    
    varied_histograms = []
    for variation in varied_weight_columns:
        entries, _ = np.histogram(
            df[fit_variable], bins=fit_bins, range=fit_range,
            weights=df[weight_column] * df[variation] / df[nominal_weight_column_for_systematic]
        )
        varied_histograms.append(entries)
    varied_histograms = np.array(varied_histograms)
    covariance = np.cov(varied_histograms, rowvar=False)
    systematics = {}
    systematics["events"] = np.histogram(df[fit_variable], bins=fit_bins, weights=df.__weight_overall__)[0]
    try:
        systematics["uncertainty"] = covariance.diagonal()**0.5
        systematics["correlation_matrix"] = covariance / np.outer(covariance.diagonal()**0.5, covariance.diagonal()**0.5)
        systematics["covariance_matrix"] = covariance
    except ValueError:
        systematics["uncertainty"] = covariance**0.5
        systematics["correlation_matrix"] = 1
        systematics["covariance_matrix"] = covariance

    return systematics


def track_finding_systematics(df, fit_variable, fit_bins, fit_range,  track_column="DTracks", tracking_eff=0.35):
    uncertainty = np.ones(len(fit_bins) - 1) * (df[track_column].mean() + 1) * tracking_eff / 100  # +1 for the signal lepton
    covariance = np.outer(uncertainty, uncertainty)
    systematics = {}
    systematics["events"] = np.histogram(df[fit_variable], bins=fit_bins, weights=df.__weight_overall__)[0]
    systematics["uncertainty"] = covariance.diagonal()**0.5
    systematics["correlation_matrix"] = covariance / np.outer(covariance.diagonal()**0.5, covariance.diagonal()**0.5)
    systematics["covariance_matrix"] = covariance
    return systematics


def pi0_finding_systematics(df, fit_variable, fit_bins, fit_range,  pi0_column="Dpi0", pi0_eff=2.):
    uncertainty = np.ones(len(fit_bins) - 1) * (df[pi0_column].mean()) * pi0_eff / 100
    covariance = np.outer(uncertainty, uncertainty)
    systematics = {}
    systematics["events"] = np.histogram(df[fit_variable], bins=fit_bins, weights=df.__weight_overall__)[0]
    systematics["uncertainty"] = covariance.diagonal()**0.5
    systematics["correlation_matrix"] = covariance / np.outer(covariance.diagonal()**0.5, covariance.diagonal()**0.5)
    systematics["covariance_matrix"] = covariance
    return systematics


def mc_statistics_systematics(df, fit_variable, fit_bins, fit_range,  weight_column="__weight_overall__"):
    
    n = df[fit_variable].values
    w = df[weight_column].values
    bin_index = np.digitize(n, fit_bins)
    # We drop the over and underflow bin here
    events = np.array([np.sum(w[np.where(bin_index == i)]) for i in range(1, len(fit_bins))])
    uncertainty = np.array([np.sqrt(np.sum(w[np.where(bin_index == i)] ** 2)) for i in range(1, len(fit_bins))])

    systematics = {}
    systematics["events"] = events
    systematics["uncertainty"] = uncertainty
    systematics["correlation_matrix"] = np.identity(len(uncertainty))
    systematics["covariance_matrix"] = np.diag(uncertainty**2)
    
    return systematics


tracks_in_D_channel = {
    31: 3,  #S D+ -> K- pi+ pi+
    32: 3,  #N D+ -> K- pi+ pi+ pi0
    33: 5,  #N D+ -> K- pi+ pi+ pi+ pi-
    34: 3,  #N D+ -> Ks pi+
    35: 3,  #N D+ -> Ks pi+ pi0
    36: 5,  #N D+ -> Ks pi+ pi+ pi-
    37: 3,  #N D+ -> Ks K^+
    38: 3,  #N D+ -> K^+ K^- \pi^+
    41: 2,  #S D0 -> K- pi+
    42: 2,  #S D0 -> K- pi+ pi0
    43: 4,  #S D0 -> K- pi+ pi+ pi-
    44: 4,  #N D0 -> K- pi+ pi+ pi- pi0
    45: 2,  #N D0 -> Ks pi0
    46: 4,  #N D0 -> Ks pi+ pi-
    47: 4,  #N D0 -> Ks pi+ pi- pi0
    48: 2,  #N D0 -> K+ K+
}

pi0_in_D_channel = {
    31: 0,  #S D+ -> K- pi+ pi+
    32: 1,  #N D+ -> K- pi+ pi+ pi0
    33: 0,  #N D+ -> K- pi+ pi+ pi+ pi-
    34: 0,  #N D+ -> Ks pi+
    35: 1,  #N D+ -> Ks pi+ pi0
    36: 0,  #N D+ -> Ks pi+ pi+ pi-
    37: 0,  #N D+ -> Ks K^+
    38: 0,  #N D+ -> K^+ K^- \pi^+
    41: 0,  #S D0 -> K- pi+
    42: 1,  #S D0 -> K- pi+ pi0
    43: 0,  #S D0 -> K- pi+ pi+ pi-
    44: 1,  #N D0 -> K- pi+ pi+ pi- pi0
    45: 1,  #N D0 -> Ks pi0
    46: 0,  #N D0 -> Ks pi+ pi-
    47: 1,  #N D0 -> Ks pi+ pi- pi0
    48: 0,  #N D0 -> K+ K+
}

kshort_in_D_channel = {
     31: 0,  #S D+ -> K- pi+ pi+
     32: 0,  #N D+ -> K- pi+ pi+ pi0
     33: 0,  #N D+ -> K- pi+ pi+ pi+ pi-
     34: 1,  #N D+ -> Ks pi+
     35: 1,  #N D+ -> Ks pi+ pi0
     36: 1,  #N D+ -> Ks pi+ pi+ pi-
     37: 1,  #N D+ -> Ks K^+
     38: 0,  #N D+ -> K^+ K^- \pi^+
     41: 0,  #S D0 -> K- pi+
     42: 0,  #S D0 -> K- pi+ pi0
     43: 0,  #S D0 -> K- pi+ pi+ pi-
     44: 0,  #N D0 -> K- pi+ pi+ pi- pi0
     45: 1,  #N D0 -> Ks pi0
     46: 1,  #N D0 -> Ks pi+ pi-
     47: 1,  #N D0 -> Ks pi+ pi- pi0
     48: 0,  #N D0 -> K+ K+
 }

slow_pion_in_Dstar_channel = {
    21: 1,  #+ D*0 -> D0 pi0
    23: 1 , #S D*+ -> D0 pi+
    24: 1,  #S D*+ -> D+ pi0
}
