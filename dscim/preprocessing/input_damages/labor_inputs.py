"""
Combine MC damages for labor
"""

import os
import time
from functools import partial
from p_tqdm import p_map
from dscim.utils.calculate_damages import concatenate_labor_damages
from dscim.menu.simple_storage import EconVars


def calculate_batch_damages(batch, ec):
    path = (
        "/shares/gcp/outputs/labor/impacts-woodwork/mc_correct_rebasing_for_integration"
    )
    save_path = "/shares/gcp/integration/float32/input_data_histclim/labor_data/new_mc/"
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
    print(f"Saved!")


if __name__ == "__main__":
    ec = EconVars(
        path_econ=f"/shares/gcp/estimation/mortality/release_2020/data/3_valuation/inputs"
    )
    # process in 3 rounds to limit memory usage
    for i in range(0, 3):
        partial_func = partial(calculate_batch_damages, ec=ec)
        print("Processing batches:")
        print(list(range(i * 5, i * 5 + 5)))
        r = p_umap(partial_func, list(range(i * 5, i * 5 + 5)))
