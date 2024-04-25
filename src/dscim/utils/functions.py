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
        weights.loc[weights.model.str.contains(f"{old}"), "model"] = (
            weights.model.apply(lambda x: x.replace(x, new))
        )
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
