import scipy.stats
import scipy.optimize
import numdifftools
import numpy as np
from uncertainties import unumpy as unp



def L(x, *pars):
    data, templates = x
    nBins = len(data)
    eta_k = [t/sum(t) for t in templates]
    assert len(pars) > 1
    nu_per_bin = [unp.nominal_values(pars[i] * eta_k[i]) for i in range(len(pars))]

    likelihood = -2 * sum(unp.nominal_values(data) * np.nan_to_num(np.log(sum(nu_per_bin)))- sum(nu_per_bin))
    if np.isposinf(likelihood):  # Catch pathological cases where L is evaluated outside of sound boundaries
        return 0
    return likelihood


def Lsys(x, 
         systematics,
         *pars):
    # templates[0] = signal
    # templates[1] = background
    data, (sig, bkg) = x
    nBins = len(data)
    nTemplates = 2
    nParamaterOfInterest = nTemplates
    nParameterOfNuisance = nTemplates * nBins
    par_interest = pars[:nParamaterOfInterest]
    par_nuisance = pars[nParamaterOfInterest:nParamaterOfInterest+nParameterOfNuisance]
    
    use_nuisance = True
    if len(par_nuisance) == 0:
        par_nuisance = np.zeros(nParameterOfNuisance)
        use_nuisance = False
        
    par_nuisance_sig = par_nuisance[:nBins]
    par_nuisance_bkg = par_nuisance[nBins:]
    
    # Debug
    #par_nuisance_sig = [x - 1 for x in par_nuisance_sig]
    
    if use_nuisance:
        sig_errors = systematics[0]["covariance"].diagonal()**0.5 / unp.nominal_values(sig)
        bkg_errors = systematics[1]["covariance"].diagonal()**0.5 / unp.nominal_values(bkg)
        sig_errors = np.nan_to_num(sig_errors)  # In case for 0 entries in template
        bkg_errors = np.nan_to_num(bkg_errors)  # In case for 0 entries in template
        fractions_sig = sig * (1 + par_nuisance_sig * sig_errors) / sum(sig * (1 + par_nuisance_sig * sig_errors))
        fractions_bkg = bkg * (1 + par_nuisance_bkg * bkg_errors) / sum(bkg * (1 + par_nuisance_bkg * bkg_errors))
    else:
        fractions_sig = sig / sum(sig)
        fractions_bkg = bkg / sum(bkg)

    eta_k = [unp.nominal_values(fractions_sig), unp.nominal_values(fractions_bkg)]
    
    nu_0_per_bin = par_interest[0] * eta_k[0]
    nu_1_per_bin = par_interest[1] * eta_k[1]
    nu_per_bin = [nu_0_per_bin, nu_1_per_bin]
    
    likelihood =  -2 * sum(unp.nominal_values(data) * np.nan_to_num(np.log(sum(nu_per_bin)))- sum(nu_per_bin))
    if use_nuisance:
        likelihood += par_nuisance_sig @ systematics[0]["inv_correlation"] @ par_nuisance_sig
        likelihood += par_nuisance_bkg @ systematics[1]["inv_correlation"] @ par_nuisance_bkg 
    if np.isposinf(likelihood):  # Catch pathological cases where L is evaluated outside of sound boundaries
        return 0
    return likelihood


def prepare_fit(templates, data_category, mc_categories):
    sample = (
        templates[data_category],
        [
            templates[category] for category in mc_categories
        ]
    )

    # x0 = [sum(unp.nominal_values(templates[category])) for category in mc_categories]
    x0 = [sum(templates[category]) for category in mc_categories]

    return sample, x0


def run_fit(L, templates, data_category, mc_categories, x0nuisance=None, calculate_covariance=True, **kwargs):
    """Run fit with a given likelihood.

    L: Likelihood function
    templates: templates_pre_fit for a given channel
    data_category: data or asimov_data (to tell which "data" to pick from the templates)
    mc_categories: template_categories (to tell which "templates" from all the templates)
    """
    sample, x0 = prepare_fit(templates, data_category, mc_categories)  
    
    if x0nuisance is not None:
        x0 = np.array([*x0, *x0nuisance])

    bounds = [(None, None) for _ in range(len(x0))]
    bounds[0] = (1, 100000)
    bounds[1] = (1, 100000)

    fit_result = scipy.optimize.minimize(
        lambda x: L(sample, *x),
        x0=unp.nominal_values(x0),
        method="SLSQP",
        bounds=bounds,
        **kwargs
    )
    fit_result.x0 = unp.nominal_values(x0)
    fit_result.covariance0 = np.diag(unp.std_devs(x0))
    fit_result.data = sum(unp.nominal_values(sample[0]))

    if calculate_covariance:
        hesse = numdifftools.Hessian(
            lambda x: L(sample, *x))(fit_result.x)
        fit_result.covariance =  np.linalg.inv(hesse / 2)
    
    return fit_result
