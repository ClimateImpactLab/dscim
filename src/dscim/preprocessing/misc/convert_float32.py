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
input_root = (
    "/shares/gcp/integration/float64/input_data_delta/ag_data/agshare_10_topcode_45p/"
)
filename_stem = ""
output_root = (
    "/shares/gcp/integration/float32/input_data_delta/ag_data/agshare_10_topcode_45p/"
)
input_dirs = glob.glob(f"{input_root}/**/", recursive=True)

convert_csv = False
convert_nc4 = True
convert_zarr = False

if not os.path.exists(output_root):
    print(f"Making {output_root}")
    os.makedirs(output_root, exist_ok=True)

for d_in in input_dirs:
    d_out = d_in.replace(input_root, output_root)
    if not os.path.exists(d_out):
        print(f"Making {d_out}")
        os.makedirs(d_out, exist_ok=True)

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


def process_nc4(f):
    df = xr.open_dataset(f)
    df = df.astype(np.float32)
    f_save = f.replace(input_root, output_root)

    try:
        df.to_netcdf(f_save)
        return f"succeeded: {f}"
    except PermissionError:
        print(f"Error saving {f}")
        return f"failed: {f}"


# if encounter error resaving - just copy over
def process_zarr(f):
    df = xr.open_mfdataset(f, engine="zarr")
    df = df.astype(np.float32)
    f_save = f.replace(input_root, output_root)
    try:
        df.to_zarr(f_save, consolidated=True, mode="w")
        return f"succeeded: {f}"
    except PermissionError:
        print(f"Error saving {f}")
        if os.path.exists(f_save):
            shutil.rmtree(f_save)
        shutil.copytree(f, f_save)
        return f"failed and copied: {f}"


def process_csv(f):
    shutil.copyfile(f, f.replace(input_root, output_root))
    return f"copied: {f}"


if convert_csv:

    i_csv = p_uimap(process_csv, files_csv)
    with open(f"{output_root}/log_csv.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_csv)


if convert_zarr:

    i_zarr = p_uimap(process_zarr, files_zarr)
    with open(f"{output_root}/log_zarr.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_zarr)

if convert_nc4:

    i_nc4 = p_uimap(process_nc4, files_nc4, num_cpus=8)
    with open(f"{output_root}/log_nc4.csv", "w", newline="") as write_file:
        write = csv.writer(write_file)
        write.writerows([r] for r in i_nc4)
