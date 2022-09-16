import pandas as pd
import xarray as xr
import numpy as np
from dscim.utils.utils import power


def ce_func(consumption, dims, eta):
    """Calculate CRRA function"""
    # use log utility when eta is 1
    if eta == 1:
        return xr.ufuncs.exp(xr.ufuncs.log(consumption).mean(dims))
    # CRRA utility otherwise
    else:
        return power(
            (power(consumption, (1 - eta)) / (1 - eta)).mean(dims) * (1 - eta),
            (1 / (1 - eta)),
        )


def mean_func(consumption, dims):
    """Calculate a mean"""
    return consumption.mean(dims)


def constant_equivalent_discount_rate(scc, marginal_damages):
    """
    Calculate the constant equivalent discount rates given a series of damages
    and the present discounted value (the SCC)
    by solving for the r in this formula:
    PDV = undiscountedcost (0) + undiscountedcost (1) / (1+r) + undiscountedcost (2) / (1+r)^2 …

    Parameters
    ----------
    scc: float
        the PDV in the equation above

    marginal_damages: 1d array
        the undiscountedcosts in the equation above

    Returns
    -------
    r: float
        the r in the equation above

    """
    # the marginal damages will be the coefficients
    coeffs = marginal_damages.copy()
    # calculate the constant term
    coeffs[0] = coeffs[0] - scc
    p = np.polynomial.Polynomial(coeffs)
    # get the roots
    roots = p.roots()
    # get rid of the complex roots
    real_roots = roots.real[abs(roots.imag) < 1e-5]
    # get rid of negative roots
    positive_real_root = real_roots[real_roots > 0]
    # if we're only left with one root we're good to go!
    assert len(positive_real_root) == 1
    # root = 1/(1+r), calculate r by inversing it
    r = 1 / positive_real_root - 1
    return r


def calculate_constant_equivalent_discount_rate(
    folder,
    recipe,
    disc,
    sel_dict,
    eta,
    rho,
    mean_dims=("simulation",),
    uncollapsed=False,
):
    """
    calls constant_equivalent_discount_rate on a set of marginal damages and sccs

    Parameters
    ----------
    folder: str
        root directory to the menu results

    recipe: str
        one of "adding_up", "risk_aversion", "equity"

    disc: str
        discount type of the scc and marginal damages that we want to derive the
        constant equivalent discount rate for

    sel_dict : dict
        dictionary of subsetting options

    uncollapsed : boolean
        if True, will use `mean` marginal damages combined with `uncollapsed_sccs` file

    Returns
    -------

    """
    # read files
    if not uncollapsed:
        sccs = xr.open_dataset(
            f"{folder}/{recipe}_{disc}_eta{eta}_rho{rho}_scc.nc4"
        ).sel(sel_dict)
        marginal_damages = xr.open_dataset(
            f"{folder}/{recipe}_{disc}_eta{eta}_rho{rho}_marginal_damages.nc4"
        ).sel(sel_dict)
    else:
        sccs = (
            xr.open_dataset(
                f"{folder}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_sccs.nc4"
            )
            .sel(sel_dict)
            .mean(mean_dims)
        )
        marginal_damages = (
            xr.open_zarr(
                f"{folder}/{recipe}_{disc}_eta{eta}_rho{rho}_uncollapsed_marginal_damages.zarr"
            )
            .load()
            .expand_dims({"fair_aggregation": ["uncollapsed"]})
            .sel(sel_dict)
            .mean(mean_dims)
        )

    # do stuff to the datasets so that they can be passed as parameters to the apply_ufunc
    sccs = (
        sccs.to_array().rename("constant_discrate").to_dataset().squeeze(dim="variable")
    )
    marginal_damages = (
        marginal_damages.to_array()
        .rename("constant_discrate")
        .to_dataset()
        .squeeze(dim="variable")
    )
    constant_discount_rates = xr.apply_ufunc(
        constant_equivalent_discount_rate,
        sccs,
        marginal_damages,
        input_core_dims=[[], ["year"]],
        vectorize=True,
    )
    return constant_discount_rates


def get_model_weights(rcp):

    # clean weights
    WEIGHT_FILE = (
        f"/shares/gcp/climate/SMME-weights/{rcp}_2090_SMME_edited_for_April_2016.tsv"
    )
    weights = pd.read_csv(
        WEIGHT_FILE, sep="\t", usecols=["quantile", "model", "weight"]
    )

    name_dict = {
        "rcp45": {
            "pattern1_": "surrogate_MRI-CGCM3_01",
            "pattern2_": "surrogate_GFDL-ESM2G_01",
            "pattern3_": "surrogate_MRI-CGCM3_06",
            "pattern5_": "surrogate_MRI-CGCM3_11",
            "pattern6_": "surrogate_GFDL-ESM2G_11",
            "pattern27_": "surrogate_GFDL-CM3_89",
            "pattern28_": "surrogate_CanESM2_89",
            "pattern29_": "surrogate_GFDL-CM3_94",
            "pattern30_": "surrogate_CanESM2_94",
            "pattern31_": "surrogate_GFDL-CM3_99",
            "pattern32_": "surrogate_CanESM2_99",
        },
        "rcp85": {
            "pattern1_": "surrogate_MRI-CGCM3_01",
            "pattern2_": "surrogate_GFDL-ESM2G_01",
            "pattern3_": "surrogate_MRI-CGCM3_06",
            "pattern4_": "surrogate_GFDL-ESM2G_06",
            "pattern5_": "surrogate_MRI-CGCM3_11",
            "pattern6_": "surrogate_GFDL-ESM2G_11",
            "pattern28_": "surrogate_GFDL-CM3_89",
            "pattern29_": "surrogate_CanESM2_89",
            "pattern30_": "surrogate_GFDL-CM3_94",
            "pattern31_": "surrogate_CanESM2_94",
            "pattern32_": "surrogate_GFDL-CM3_99",
            "pattern33_": "surrogate_CanESM2_99",
        },
    }

    common = {
        "access1-0": "ACCESS1-0",
        "bnu-esm": "BNU-ESM",
        "canesm2": "CanESM2",
        "ccsm4": "CCSM4",
        "cesm1-bgc": "CESM1-BGC",
        "cnrm-cm5": "CNRM-CM5",
        "csiro-mk3-6-0": "CSIRO-Mk3-6-0",
        "gfdl-cm3": "GFDL-CM3",
        "gfdl-esm2g": "GFDL-ESM2G",
        "gfdl-esm2m": "GFDL-ESM2M",
        "ipsl-cm5a-lr": "IPSL-CM5A-LR",
        "ipsl-cm5a-mr": "IPSL-CM5A-MR",
        "miroc-esm-chem": "MIROC-ESM-CHEM",
        "miroc-esm*": "MIROC-ESM",
        "miroc5": "MIROC5",
        "mpi-esm-lr": "MPI-ESM-LR",
        "mpi-esm-mr": "MPI-ESM-MR",
        "mri-cgcm3": "MRI-CGCM3",
        "noresm1-m": "NorESM1-M",
    }

    [v.update(common) for k, v in name_dict.items()]

    for old, new in name_dict[rcp].items():
        weights.loc[
            weights.model.str.contains(f"{old}"), "model"
        ] = weights.model.apply(lambda x: x.replace(x, new))
    weights.model = weights.model.apply(lambda x: x.replace("*", ""))
    weights = weights.rename(columns={"model": "gcm"}).set_index("gcm").to_xarray()
    weights = weights.assign_coords({"rcp": rcp})

    return weights.weight


def gcms():
    return [
        "ACCESS1-0",
        "CCSM4",
        "GFDL-CM3",
        "IPSL-CM5A-LR",
        "MIROC-ESM-CHEM",
        "bcc-csm1-1",
        "CESM1-BGC",
        "GFDL-ESM2G",
        "IPSL-CM5A-MR",
        "MPI-ESM-LR",
        "BNU-ESM",
        "CNRM-CM5",
        "GFDL-ESM2M",
        "MIROC5",
        "MPI-ESM-MR",
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",
        "MIROC-ESM",
        "MRI-CGCM3",
        "NorESM1-M",
        "surrogate_GFDL-CM3_89",
        "surrogate_GFDL-ESM2G_11",
        "surrogate_CanESM2_99",
        "surrogate_GFDL-ESM2G_01",
        "surrogate_MRI-CGCM3_11",
        "surrogate_CanESM2_89",
        "surrogate_GFDL-CM3_94",
        "surrogate_MRI-CGCM3_01",
        "surrogate_CanESM2_94",
        "surrogate_GFDL-CM3_99",
        "surrogate_MRI-CGCM3_06",
        "surrogate_GFDL-ESM2G_06",
    ]


def US_territories():
    return [
        "USA",
        "XBK",
        "GUM",
        "XHO",
        "XJV",
        "XJA",
        "XKR",
        "XMW",
        "XNV",
        "MNP",
        "XPL",
        "PRI",
        "VIR",
        "XWK",
    ]
