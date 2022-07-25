import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from uncertainties import ufloat
from uncertainties import unumpy as unp

def plot_correlation_matrix(
    matrix, 
    bin_edges, 
    fit_variable_label, 
    fit_variable_unit, 
    channel_label
    ):
    fig, ax = plt.subplots(dpi=130, figsize=(6.4, 4.4 / 0.8))
    im = ax.imshow(matrix, vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(bin_edges)) - 0.5)
    ax.set_yticks(np.arange(len(bin_edges)) - 0.5)
    # ... and label them with the respective list entries
    ax.set_xticklabels([f"{x:.2f}" for x in bin_edges])
    ax.set_yticklabels([f"{x:.2f}" for x in bin_edges])

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    #cbar.ax.set_ylabel("...", rotation=-90, va="center")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="major", color="black", linestyle='-', linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_xlabel(f"{fit_variable_label} / ({fit_variable_unit})")
    ax.set_ylabel(f"{fit_variable_label} / ({fit_variable_unit})")
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(bin_edges)-1):
        for j in range(len(bin_edges)-1):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize=6)


    add_watermark(ax, channel_label)
    plt.tight_layout()
    plt.show()
    plt.close()
    

def chi2(df1, df2, var, variables):

    bins = np.linspace(variables[var][1][0], variables[var][1][1], num=variables[var][0]+1)

    def get_expectation_and_uncertainty_sq(df, bins):
        expectation = df.groupby(pd.cut(df[var], bins))["__plotting_weight__"].aggregate(np.sum)

        df["__plotting_weight_sq__"] = df["__plotting_weight__"].apply(lambda x: x**2)
        uncertainty_sq = df.groupby(pd.cut(df[var], bins))["__plotting_weight_sq__"].aggregate(np.sum)
        return expectation, uncertainty_sq

    df1_expectation, df1_uncertainty_sq = get_expectation_and_uncertainty_sq(df1, bins)
    df2_expectation, df2_uncertainty_sq = get_expectation_and_uncertainty_sq(df2, bins)
    return np.sum(np.nan_to_num(df1_expectation - df2_expectation)**2 / (df1_uncertainty_sq + df2_uncertainty_sq))


def init_plot_style():
    """Define the rcParams for the plot. Requires matplotlib to be imported as mpl."""
    my_rc_params = {
        "xtick.direction": "out",
        "xtick.major.size": 8.0,
        "xtick.minor.size": 4.0,
        "xtick.minor.visible": True,
        "xtick.major.width": 1.2,
        "xtick.minor.width": 0.9,
        "ytick.direction": "out",
        "ytick.major.size": 8.0,
        "ytick.minor.size": 4.0,
        "ytick.minor.visible": True,
        "ytick.major.width": 1.2,
        "ytick.minor.width": 0.9,
        "errorbar.capsize": 0,
        "axes.linewidth": 1.2,
        # "font.familiy": "serif",
        "font.size": 12,
        "axes.grid": False,
        "ytick.right": False,
        "xtick.top": False
    }
    for key in my_rc_params.keys():
        mpl.rcParams[key] = my_rc_params[key]


class Tango():
    """Class containing class members for the Tango colour palette."""
    scarlet_red_light = '#ef2929'
    scarlet_red = '#cc0000'
    scarlet_red_dark = '#a40000'

    aluminium_light = '#eeeeec'
    aluminium = '#d3d7cf'
    aluminium_dark = '#babdb6'

    butter_light = '#fce94f'
    butter = '#edd400'
    butter_dark = '#c4a000'

    chameleon_light = '#8ae234'
    chameleon = '#73d216'
    chameleon_dark = '#4e9a06'

    orange_light = '#fcaf3e'
    orange = '#f57900'
    orange_dark = '#ce5c00'

    chocolate_light = '#e9b96e'
    chocolate = '#c17d11'
    chocolate_dark = '#8f5902'

    sky_blue_light = '#729fcf'
    sky_blue = '#3465a4'
    sky_blue_dark = '#204a87'

    plum_light = '#ad7fa8'
    plum = '#75507b'
    plum_dark = '#5c3566'

    slate_light = '#888a85'
    slate = '#555753'
    slate_dark = '#2e3436'


color_dict = {
    9: Tango.slate,
    8: Tango.sky_blue,
    7: Tango.aluminium_dark,
    6: Tango.plum_dark,
    5: Tango.plum_light,
    4.1: Tango.scarlet_red_light,
    4.2: Tango.scarlet_red_dark,
    3: Tango.orange,
    #3.1: Tango.orange_light,
    #3.2: Tango.orange,
    #2: Tango.chameleon_dark,
    #1: Tango.chameleon_light,
}


channel_label = {
    11: r"$B^0 \to D^+ e \nu_e$",
    12: r"$B^0 \to D^+ \mu \nu_\mu$",
    13: r"$B^+ \to D^0 e \nu_e$",
    14: r"$B^+ \to D^0 \mu \nu_\mu$",
    15: r"$B^0 \to D^{*+} e \nu_e$",
    16: r"$B^0 \to D^{*+} \mu \nu_\mu$",
    17: r"$B^+ \to D^{*0} e \nu_e$",
    18: r"$B^+ \to D^{*0} \mu \nu_\mu$",
    21: r"$D^{*0} \to D^0 \pi^0$",
    22: r"$D^{*0} \to D^0 \gamma$",
    23: r"$D^{*+} \to D^0 \pi^+$",
    24: r"$D^{*+} \to D^+ \pi^0$",
    31: r"$D^+ \to K^- \pi^+ \pi^+$",
    32: r"$D^+ \to K^- \pi^+ \pi^+ \pi^0$",
    33: r"$D^+ \to K^- \pi^+ \pi^+ \pi^+ \pi^-$",
    34: r"$D^+ \to K_\mathrm{S}^0 \pi^+$",
    35: r"$D^+ \to K_\mathrm{S}^0 \pi^+ \pi^0$",
    36: r"$D^+ \to K_\mathrm{S}^0 \pi^+ \pi^+ \pi^-$",
    37: r"$D^+ \to K_\mathrm{S}^0 K^+$",
    38: r"$D^+ \to K^+ K^- \pi^+$",
    41: r"$D^0 \to K^- \pi^+$",
    42: r"$D^0 \to K^- \pi^+ \pi^0$",
    43: r"$D^0 \to K^- \pi^+ \pi^+ \pi^-$",
    44: r"$D^0 \to K^- \pi^+ \pi^+ \pi^- \pi^0$",
    45: r"$D^0 \to K_\mathrm{S}^0 \pi^0$",
    46: r"$D^0 \to K_\mathrm{S}^0 \pi^+ \pi^-$",
    47: r"$D^0 \to K_\mathrm{S}^0 \pi^+ \pi^- \pi^0$",
    48: r"$D^0 \to K^- K^+$",
    (11, 12): r"$B^0 \to D^+ \ell \nu_\ell$",
    (13, 14): r"$B^+ \to D^0 \ell \nu_\ell$",
    (15, 16): r"$B^0 \to D^{*+} \ell \nu_\ell$",
    (17, 18): r"$B^+ \to D^{*0} \ell \nu_\ell$",
    "all": r"$B \to D^{*} \ell \nu_\ell$",
}


from typing import Dict, List, Optional, Sequence, Tuple, Union

FigType = mpl.figure.Figure
AxesType = mpl.axes.Axes
FigAxesTuple = Tuple[FigType, AxesType]
SigTypes = List[int]
SigIDDict = Dict[str, int]
SigLabelDict = Dict[str, str]
ColorDict = Dict[str, str]

STD_FIGSIZE = (6.4 * 0.9, 4.4 * 0.9)
STD_DPI = 130
STD_HISTYSCALE = 1.45


def get_bin_mids(bin_edges: np.array) -> np.array:
    # TODO: Docstring
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def get_bin_width(bin_edges: np.array) -> np.array:
    # TODO: Docstring
    return bin_edges[1] - bin_edges[0]


def get_bin_non_equidistant_width(bin_edges: np.array) -> np.array:
    # TODO: Docstring
    return bin_edges[1:] - bin_edges[:-1]


def set_yscale(
    ax: mpl.axes.Axes, scale_factor: float = STD_HISTYSCALE
) -> mpl.axes.Axes:
    # TODO: Docstring
    ylow, yhigh = ax.get_ylim()
    ax.set_ylim(ylow, scale_factor * yhigh)
    return ax


def sci_notation(number: float, sig_fig: int = 2) -> str:
    """Return number as formatted string using `x10⁻¹` superscript
    scientific notation instead `e-1`.


    :param number: Number to format
    :type number: float
    :param sig_fig: Significant decimals to round to, defaults to 2
    :type sig_fig: int, optional
    :return: Formatted string
    :rtype: str
    """
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)  # removed leading "+" and strips leading zeros too.

    super_scriptinate = (
        lambda x: x.replace("0", "⁰")
        .replace("1", "¹")
        .replace("2", "²")
        .replace("3", "³")
        .replace("4", "⁴")
        .replace("5", "⁵")
        .replace("6", "⁶")
        .replace("7", "⁷")
        .replace("8", "⁸")
        .replace("9", "⁹")
        .replace("-", "⁻")
    )

    return a + r"$\times$ 10" + super_scriptinate(str(b))


def create_ylabel(y_str: str, bin_width: float, unit: Optional[str] = None) -> str:
    try:
        if bin_width < 0.01:
            bin_width = sci_notation(bin_width)
        else:
            bin_width = str(np.around(bin_width, 2))
        if unit:
            y_label = f"{y_str} / ({bin_width} {unit})"
        else:
            y_label = f"{y_str} / ({bin_width})"

        return y_label
    except:
        y_label = y_str
        return y_label


def plot_mc_uncertainty_band(
    ax: AxesType,
    bin_counts: np.array,
    bin_mids: np.array,
    bin_width: float,
    uncertainty: np.array,
):
    ax.bar(
        x=bin_mids,
        height=2 * uncertainty,
        width=bin_width,
        bottom=bin_counts - uncertainty,
        color="black",
        hatch="///////",
        fill=False,
        lw=0,
        label="Uncertainty",
    )


def add_lumi(
    ax: AxesType,
    l: float,
    px: float = 0.700,
    py: float = 1.022, #0.839,
    fontsize: str = 12,
    *args,
    **kwargs,
):
    lumi_string = f"$\int\,L\,\mathrm{{dt}}\;=\;{l:.0f}\; \mathrm{{fb}}^{{-1}}$"
    ax.text(
        px, py, lumi_string, fontsize=fontsize, transform=ax.transAxes, *args, **kwargs
    )


def add_watermark(
    ax: AxesType,
    t: str = None,
    logo: str = "Belle",
    px: float = 0.033,
    py: float = 1.022, #0.915,
    fontsize: int = 12,
    alpha_logo=1,
    shift: float = 0.15,
    bstyle: str = "normal",
    *args,
    **kwargs,
):

    ax.text(
        px,
        py,
        logo,
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
        style=bstyle,
        alpha=alpha_logo,
        weight="bold",
        *args,
        **kwargs,
    )
    ax.text(
        px + shift,
        py,
        t,
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
        alpha=alpha_logo,
        *args,
        **kwargs,
    )


def add_channel(
    ax: AxesType,
    t: str = None,
    px: float = 0.033,
    py: float = 0.915,
    fontsize: int = 12,
    alpha_logo=1,
    shift: float = 0.15,
    bstyle: str = "normal",
    weight: str = "bold",
    *args,
    **kwargs,
):

    ax.text(
        px,
        py,
        t,
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
        style=bstyle,
        alpha=alpha_logo,
        weight=weight,
        *args,
        **kwargs,
    )


def calculate_ratio(
    numerator: np.array,
    denominator: np.array,
    numerator_uncertainty: Optional[np.array] = None,
    denominator_uncertainty: Optional[np.array] = None,
):
    numerator_uncertainty = (
        np.zeros_like(numerator)
        if numerator_uncertainty is None
        else numerator_uncertainty
    )
    denominator_uncertainty = (
        np.zeros_like(denominator)
        if denominator_uncertainty is None
        else denominator_uncertainty
    )

    uh_numerator = unp.uarray(np.nan_to_num(numerator), np.nan_to_num(numerator_uncertainty))
    uh_denominator = unp.uarray(denominator, denominator_uncertainty)
    uh_denominator[unp.nominal_values(uh_denominator) == 0] = ufloat(1e-7, 1e-14)
    #print(uh_numerator)
    #print(uh_denominator)

    return uh_numerator / uh_denominator


def create_hist_ratio_figure(
    fig_size=(6.4, 4.4 / 0.8), height_ratio=(4, 1), dpi=130
):
    return plt.subplots(
        nrows=2,
        ncols=1,
        figsize=fig_size,
        dpi=dpi,
        sharex="all",
        gridspec_kw={"height_ratios": [height_ratio[0], height_ratio[1]]},
    )


def data_vs_mc_stacked_hist(
    mc_df: pd.DataFrame,
    data_df: pd.DataFrame,
    column: str,
    sig_types: SigTypes,
    bins: Optional[Union[int, Sequence, str]] = None,
    xrange: Optional[Tuple[float, float]] = None,
    weight_column: Optional[str] = None,
    sig_id_var: str = "__sigID__",
    sig_labels: Optional[SigLabelDict] = None,
    color_dict: Optional[ColorDict] = None,
    y_str: str = "Entries",
    var_str: Optional[str] = None,
    unit: Optional[str] = None,
    equidistant_bins=True,
) -> Tuple[FigType, Tuple[AxesType, AxesType]]:

    fig, ax = create_hist_ratio_figure()
    fig.subplots_adjust(hspace=0.1)

    data = [
        mc_df.query(f"{sig_id_var} == {sig_type}")[column]
        for sig_type in sig_types
    ]

    if weight_column:
        weights = [
            mc_df.query(f"{sig_id_var} == {sig_type}")[weight_column]
            for sig_type in sig_types
        ]
    else:
        weights = None

    labels = [sig_labels[sig_type] for sig_type in sig_types] if sig_labels else None
    colors = [color_dict[sig_type] for sig_type in sig_types] if color_dict else None

    bc, be, _ = ax[0].hist(
        data,
        weights=weights,
        bins=bins,
        range=xrange,
        label=labels,
        stacked=True,
        color=colors,
        histtype="stepfilled",
        edgecolor="black",
        lw=0.5,
    )

    bin_mids = get_bin_mids(be)
    bin_width = (
        get_bin_width(be) if equidistant_bins else get_bin_non_equidistant_width(be)
    )

#    print("Calculating covariance matrices for histogram components..")
#
#    component_stat_cov_mats = {}
#    component_pid_cov_mats = {}
#    component_fei_cov_mats = {}
#
#    for sig_type in sig_types:
#        component = mc_df.query(f"{sig_id_var} == {sig_type}")
#
#        component_stat_cov_mats[sig_type] = get_statistical_cov_mat(
#            component[column], weights=component[weight_column], bins=be
#        )
#
#        component_fei_cov_mats[sig_type] = get_cov_mat_from_up_down_variation(
#            component[column],
#            bins=be,
#            weights=component[weight_column],
#            nominal_weights=component.__feiCorrectionWeight__,
#            up_weights=component.__feiCorrectionWeightVarUp__,
#            down_weights=component.__feiCorrectionWeightVarDown__,
#        )
#
#        varied_hists = list()
#        for variation_weight in [f"LeptonIDWeight_{i}" for i in range(0, 50)]:
#            variation_weight_columns = [
#                "__weight__",
#                "__lumiWeight__",
#                "__sampleSplitWeightGroupsMVASample__",
#                "__sampleSplitWeightGroupsSubSample__",
#                "__feiCorrectionWeight__",
#                variation_weight,
#            ]
#            component.loc[:, "__scalingWeightVariedTemp__"] = component.loc[
#                :, variation_weight_columns
#            ].product(axis=1)
#            varied_hists.append(
#                np.histogram(
#                    component[column],
#                    weights=component["__scalingWeightVariedTemp__"],
#                    bins=be,
#                )[0]
#            )
#
#        component_pid_cov_mats[sig_type] = np.cov(np.array(varied_hists), rowvar=False)
#
#    bin_by_bin_uncertainty = np.sqrt(
#        np.diag(
#            sum_cov_mat_dict(component_stat_cov_mats)
#            + sum_cov_mat_dict(component_fei_cov_mats)
#            + sum_cov_mat_dict(component_pid_cov_mats)
#        )
#    )

    bc, _ = np.histogram(mc_df[column], bins=be, weights=mc_df[weight_column])

#    plot_mc_uncertainty_band(
#        ax=ax[0],
#        bin_counts=bc,
#        bin_mids=bin_mids,
#        bin_width=bin_width,
#        uncertainty=bin_by_bin_uncertainty,
#    )

    data_bc, _ = np.histogram(data_df[column], bins=be)

    ax[0].errorbar(
        x=bin_mids,
        y=data_bc,
        yerr=np.sqrt(data_bc),
        ls="",
        marker=".",
        color="black",
        label="Data",
    )
    set_yscale(ax[0])
    ax[0].legend(frameon=False, fontsize="x-small", ncol=3, loc='upper right')

    if equidistant_bins:
        ax[0].set_ylabel(create_ylabel(y_str, bin_width, unit))
    else:
        ax[0].set_ylabel(y_str)

    var_str = var_str if var_str else column
    x_label = var_str + f" [{unit}]" if unit else var_str

    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(r"Data/MC")
    ax[1].set_axisbelow(True)
    ax[1].grid(axis="y")

    mc_bc = bc
    mc_bc_plus_unc = bc #+ bin_by_bin_uncertainty
    ratio = calculate_ratio(numerator=mc_bc_plus_unc, denominator=mc_bc)
    ax[1].bar(
        x=bin_mids,
        height=2 * (unp.nominal_values(ratio) - 1),
        width=bin_width,
        bottom=(1 - (unp.nominal_values(ratio) - 1)),
        color="grey",
        alpha=0.8,
    )

    ratio = calculate_ratio(
        numerator=data_bc,
        denominator=mc_bc,
        numerator_uncertainty=np.sqrt(data_bc),
        denominator_uncertainty=None #bin_by_bin_uncertainty,
    )

    ax[1].errorbar(
        get_bin_mids(be),
        unp.nominal_values(ratio),
        yerr=unp.std_devs(ratio),
        ls="",
        marker=".",
        color="black",
    )
    ax[1].set_ylim(bottom=0.6, top=1.4)

    return fig, ax



