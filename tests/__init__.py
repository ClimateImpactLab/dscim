import os
import dask
import zipfile
import pandas as pd
import xarray as xr

from pathlib import Path

dask.config.set(scheduler="single-threaded")


def open_example_dataset(name, *args, **kwargs):
    ext = Path(name).suffix

    if ext == ".nc4":
        ds = xr.open_dataset(
            os.path.join(os.path.dirname(__file__), "data", "menu_results", name),
            *args,
            **kwargs,
        )
    elif ext == ".csv":
        ds = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "menu_results", name),
            *args,
            **kwargs,
        )

    return ds


def open_zipped_results(name, *args, **kwargs):
    unzip = zipfile.ZipFile(
        os.path.join(os.path.dirname(__file__), "data", "menu_results.zip")
    )
    ext = Path(name).suffix

    if ext == ".nc4":
        with unzip.open(f"menu_results/{name}") as file:
            ds = xr.load_dataset(file, *args, **kwargs)

    elif ext == ".csv":
        with unzip.open(f"menu_results/{name}") as file:
            ds = pd.read_csv(file, *args, **kwargs)

    return ds
