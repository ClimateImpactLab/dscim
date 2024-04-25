import pandas as pd
import xarray as xr
import numpy as np
from dscim.utils.utils import power


def ce_func(consumption, dims, eta):
    """Calculate CRRA function"""
    # use log utility when eta is 1
    if eta == 1:
        return np.exp(np.log(consumption).mean(dims))
    # CRRA utility otherwise
    else:
        return power(
            (power(consumption, (1 - eta)) / (1 - eta)).mean(dims) * (1 - eta),
            (1 / (1 - eta)),
        )


def mean_func(consumption, dims):
    """Calculate a mean"""
    return consumption.mean(dims)


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
