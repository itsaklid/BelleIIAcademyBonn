"""Definitions of the decay id's: 

B_decay_label_to_decay_id = {
    "B0:D_e": 11,
    "B0:D_mu": 12,
    "B-:D0_e": 13,
    "B-:D0_mu": 14,
    "B0:D*_e": 15,
    "B0:D*_mu": 16,
    "B-:D*0_e": 17,
    "B-:D*0_mu": 18,
}

Dstar_decay_label_to_decay_id = {
    "D*0:D0pi0": 21,
    "D*0:D0gamma": 22,
    "D*+:D0pi+": 23,
    "D*+:D+pi0": 24,
}

"""

sig_id_to_label = {
    3: '$D \\ell \\nu$',
    5: '$D^{**}(\\rightarrow D^{(*)} \\pi^0) \\ell \\nu$',
    6: '$D^{**}(\\rightarrow D^{(*)} \\pi^\\pm) \\ell \\nu$',
    7: 'Hadronic Bkg',
    8: '$B\\bar{B}$ Bkg',
    9: 'Continuum'
}
sig_id_to_label[4.1] = '$D^* \\ell \\nu$ (correct $\pi_\mathrm{slow}$)'
sig_id_to_label[4.2] = '$D^* \\ell \\nu$ (wrong $\pi_\mathrm{slow}$)'

