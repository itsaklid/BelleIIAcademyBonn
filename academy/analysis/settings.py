import numpy as np

bin_edges_wReco = np.array([*np.linspace(1.0, 1.45, num=10), 2])
bin_edges_costhetalReco = np.array([*np.linspace(-1, 1, num=11)])
bin_edges_costhetavReco = np.array([*np.linspace(-1, 1, num=11)])
bin_edges_chi_Reco = np.array([*np.linspace(0, 2*np.pi, num=11)])

bin_edges_all = {
    "wReco": bin_edges_wReco,
    "costhetalReco": bin_edges_costhetalReco,
    "costhetavReco": bin_edges_costhetavReco,
    "chi_Reco": bin_edges_chi_Reco,
}

voi_labels = {
    "wReco": r"$w$",
    "costhetalReco": r"$\cos \theta_\ell$",
    "costhetavReco": r"$\cos \theta_V$",
    "chi_Reco": r"$\chi$",
}

voi_units = {
    "wReco": None,
    "costhetalReco": None,
    "costhetavReco": None,
    "chi_Reco": r"rad",
}

voi_mc = {
    "wReco": "wMC",
    "costhetalReco": "costhetalMC",
    "costhetavReco": "costhetavMC",
    "chi_Reco": "chi_MC",
}

