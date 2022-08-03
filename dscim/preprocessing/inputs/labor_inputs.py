"""
Combine MC damages for labor
"""

import os
import time
from functools import partial
from p_tqdm import p_umap
from dscim.utils.calculate_damages import concatenate_labor_damages
from dscim.menu.simple_storage import EconVars

print("testing message: version jun 25")


def calculate_batch_damages(batch, ec, input_path, output_path):
    path = input_path
    save_path = output_path
    print(f"Processing batch={batch} damages in {os.getpid()}")
    concatenate_labor_damages(
        input_path=path,
        save_path=save_path,
        ec_cls=ec,
        variable="rebased",
        val_type="wage-levels",
        format_file="zarr",
        query=f"exists==True&batch=='batch{batch}'",
    )
    print("Saved!")


def labor_inputs(
    path_econ="/shares/gcp/estimation/mortality/release_2020/data/3_valuation/inputs",
    input_path="/shares/gcp/outputs/labor/impacts-woodwork/mc_correct_rebasing_for_integration",
    output_path="/shares/gcp/integration/float32/input_data_histclim/labor_data/new_mc/",
):
    # if __name__ == "__main__":
    ec = EconVars(path_econ=path_econ)
    # process in 3 rounds to limit memory usage
    for i in range(0, 3):
        partial_func = partial(
            calculate_batch_damages,
            ec=ec,
            input_path=input_path,
            output_path=output_path,
        )
        print("Processing batches:")
        print(list(range(i * 5, i * 5 + 5)))
        p_umap(partial_func, list(range(i * 5, i * 5 + 5)))
