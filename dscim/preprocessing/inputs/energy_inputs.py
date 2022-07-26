"""
Run valuations for energy
"""

import os
import time
from functools import partial
from p_tqdm import p_umap
from dscim.utils.calculate_damages import concatenate_energy_damages
from dscim.menu.simple_storage import EconVars
from dscim.preprocessing.input_damages import read_energy_files_parallel

print("testing message: version jun 25")


def calculate_batch_damages(batch, ec):
    path = "/shares/gcp/outputs/energy_pixel_interaction/impacts-blueghost/integration_resampled"
    save_path = (
        "/shares/gcp/integration/float32/input_data_histclim/energy_data/hybrid_price/"
    )
    print(f"Processing batch={batch} damages in {os.getpid()}")
    concatenate_energy_damages(
        input_path=path,
        file_prefix="TINV_clim_integration_total_energy",
        save_path=save_path,
        ec_cls=ec,
        variable="rebased",
        format_file="zarr",
        query=f"exists==True&batch=='batch{batch}'",
    )
    print("Saved!")


def energy_inputs(
    re_calculate=False,
    path_econ="/shares/gcp/estimation/mortality/release_2020/data/3_valuation/inputs",
    input_path="/shares/gcp/outputs/energy_pixel_interaction/impacts-blueghost/integration_resampled",
    output_path="/shares/gcp/integration/float32/input_data_histclim/energy_data/",
):
    # if __name__ == "__main__":

    # re_calculate = False
    ec = EconVars(path_econ)

    if re_calculate:
        read_energy_files_parallel(
            input_path=input_path,
            save_path=output_path,
            ec_cls=ec,
            seed="TINV_clim_integration_total_energy_delta",
        )

        read_energy_files_parallel(
            input_path=input_path,
            save_path=output_path,
            ec_cls=ec,
            seed="TINV_clim_integration_total_energy_histclim",
        )

    # process in 3 rounds to limit memory usage
    for i in range(0, 3):
        partial_func = partial(calculate_batch_damages, ec=ec)
        print("Processing batches:")
        print(list(range(i * 5, i * 5 + 5)))
        r = p_umap(partial_func, list(range(i * 5, i * 5 + 5)))
