# Pass a directory, output log files detailing number of nans each file has

import os
import numpy as np
import pandas as pd
import xarray as xr
import glob
import shutil
import csv
from p_tqdm import p_uimap


# NOTE: modify paths in this cell
# NOTE: include the trailing / in the root paths
input_root = "/shares/gcp/integration/float32/sectoral_ir_damages/mortality_epa_vsl/"

filename_stem = ""
output_root = "/shares/gcp/integration/float32/sectoral_ir_damages/mortality_epa_vsl/"

check_csv = False
check_nc4 = False
check_zarr = False

# grab all the files
files_nc4 = (
    glob.glob(f"{input_root}/**/*.nc", recursive=True)
    + glob.glob(f"{input_root}/*.nc", recursive=True)
    + glob.glob(f"{input_root}/**/*.nc4", recursive=True)
    + glob.glob(f"{input_root}/*.nc4", recursive=True)
)

files_zarr = glob.glob(f"{input_root}/**/*.zarr/", recursive=True) + glob.glob(
    f"{input_root}/*.zarr/", recursive=True
)

files_csv = glob.glob(f"{input_root}/**/*.csv", recursive=True) + glob.glob(
    f"{input_root}/*.csv", recursive=True
)

regions = [
    "ARG.8.244",
    "ATA",
    "ATF.R3ad2a7b0834665e6",
    "AUS.1.1",
    "AUS.10.1145",
    "AUS.11.1345",
    "AUS.3.112",
    "AUS.5.387",
    "AUS.5.400",
    "AUS.6.687",
    "AUS.7.989",
    "AUS.7.995",
    "BRA.8.836.1993",
    "BRA.8.837.1994",
    "BVT",
    "CAN.11.269.4448",
    "CAN.2.42.1074",
    "CAN.3.58.1374",
    "CAN.9.148.Rd02357429ca755ba",
    "CHN.21.226.1500",
    "CHN.30.318.2210.R55b41404d256c30a",
    "CHN.30.318.2210.Rdb1a80fb7e65ef11",
    "CL-",
    "COL.26.852",
    "DOM.19.91",
    "ESP.6.27.191.4867",
    "ESP.6.27.192.4868",
    "GRL.1.2",
    "GRL.2.9",
    "GRL.3.18",
    "HMD",
    "IDN.14.203",
    "IND.12.129.431",
    "IND.12.129.432",
    "IND.12.132.460",
    "IND.12.134.487",
    "IND.2.17.134",
    "IND.2.17.135",
    "IND.2.17.136",
    "IND.2.17.138",
    "IND.2.17.142",
    "IND.2.17.143",
    "IOT",
    "JPN.37.1512",
    "MRT.12.38",
    "NZL.10.42.253",
    "PER.21.169.1647",
    "SGS",
    "SJM.1",
    "SP-",
    "TWN.2.2",
    "ZAF.9.313",
]


def fill_known_nans(ds, path):
    if ("rcp" in ds.dims) and ("ssp" in ds.dims):
        ds = xr.where((ds.ssp == "SSP1") & (ds.rcp == "rcp85"), 0, ds)
        ds = xr.where((ds.ssp == "SSP5") & (ds.rcp == "rcp45"), 0, ds)
    if ("rcp" in ds.dims) and ("gcm" in ds.dims):
        ds = xr.where(
            (ds.gcm == "surrogate_GFDL-ESM2G_06") & (ds.rcp == "rcp45"), 0, ds
        )
    if ("agriculture" in path) or ("ag_data" in path):
        print("replacing ag-only missings")
        ds = xr.where((ds.gcm == "ACCESS1-0") & (ds.rcp == "rcp85"), 0, ds)
    if ("region" in ds.dims) and ("gcm" in ds.dims):
        ds = xr.where(ds.region.isin(regions), 0, ds)
    return ds


def check_nc4_nans(path):
    print(path)
    try:
        ds = xr.open_dataset(path)
        ds = fill_known_nans(ds, path)
        out = path + "   " + str(ds.isnull().sum().data_vars)
        ds.close()
        return out
    except FileNotFoundError:
        print(f"Error checking {path}")
        return f"failed: {path}"


def check_zarr_nans(path):
    print(path)
    try:
        ds = xr.open_zarr(path, chunks=None)
        ds = fill_known_nans(ds, path)
        out = path + "   " + str(ds.isnull().sum().data_vars)
        ds.close()
        return out
    except FileNotFoundError:
        print(f"Error checking {path}")
        return f"failed: {path}"


def check_csv_nans(path):
    print(path)
    try:
        ds = pd.read_csv(path)
        output = path + "   " + str(ds.isnull().sum().sum())
        return output
    except FileNotFoundError:
        print(f"Error checking {path}")
        return f"failed: {path}"


check_zarr_nans(
    "/shares/gcp/integration/float32/sectoral_ir_damages/mortality_epa_vsl/impacts-darwin-montecarlo-damages-vsl_popavg-histclim-delta.zarr"
)


if check_csv:
    print("check csv++++++++++")
    i_csv = p_uimap(check_csv_nans, files_csv)
    with open(f"{output_root}/log_csv.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_csv)


if check_zarr:
    print("check zarr++++++++++")
    #     i_zarr = []
    #     for f in files_zarr:
    #         i_zarr.append(check_zarr_nans(f))
    i_zarr = p_uimap(check_zarr_nans, files_zarr, num_cpus=1)
    with open(f"{output_root}/log_zarr.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_zarr)

if check_nc4:
    print("check nc4++++++++++")
    i_nc4 = p_uimap(check_nc4_nans, files_nc4, num_cpus=1)
    with open(f"{output_root}/log_nc4.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_nc4)


# NOTE: code snippet from Ian for checking missingness

# def check_finished_zarr_workflow(
#     finalstore=None,
#     tmpstore=None,
#     varname=None,
#     final_selector={},
#     check_final=True,
#     check_temp=True,
# ):
#     finished = False
#     temp = False
#     if check_final:
#         finished = (
#             xr.open_zarr(finalstore, chunks=None)[varname]
#             .sel(final_selector, drop=True)
#             .notnull()
#             .all()
#             .item()
#         )
#     if finished:
#         return True
#     if check_temp:
#         if tmpstore.fs.isdir(tmpstore.root):
#             try:
#                 temp = xr.open_zarr(tmpstore, chunks=None)
#                 if (
#                     varname in temp.data_vars
#                     and "year" in temp.dims
#                     and len(temp.year) > 0
#                 ):
#                     finished = temp[varname].notnull().all().item()
#             except Exception:
#                 ...
#     return finished
