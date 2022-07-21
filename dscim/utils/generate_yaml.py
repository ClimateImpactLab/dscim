import os
import sys
import yaml
import dscim
from dscim import Waiter

PKG_ROOT = os.path.dirname(os.path.dirname(dscim.__file__))
assert "dscim" in os.listdir(PKG_ROOT)
assert "menu" in os.listdir(os.path.join(PKG_ROOT, "dscim"))


def configure_dscim(root_dir, config_name="config", strict_combos=True):
    """This function writes a config file, which is then used to
    configure a `Waiter` object which can be used to run the `dscim`
    classes.

    Parameters
    ----------
    root_dir: str
        A filepath corresponding to the root directory of the downloaded data
    """

    class_dict = {
        "global_parameters": {
            "fair_aggregation": [
                "median_params",
            ]
        },
        "econvars": {"path_econ": f"{root_dir}/econvars/"},
        "climate": {
            "gases": ["CO2_Fossil"],
            "gmsl_path": f"{root_dir}/climate/coastal_gmsl_sims.zarr",
            "gmst_path": f"{root_dir}/climate/GMTanom_all_temp_2001_2010_smooth.csv",
            "gmst_fair_path": f"{root_dir}/climate/fair_gmst.nc",
            "gmsl_fair_path": f"{root_dir}/climate/fair_gmsl.zarr",
            "damages_pulse_conversion_path": f"{root_dir}/climate/fair_CO2_Fossil-CH4-N2O_conversion_factors.nc",
            "pulse_year": 2020,
            "emission_scenarios": ["ssp370", "ssp245", "ssp460"],
        },
        "sectors": {
            "combined": {
                "sector_path": None,
                "save_path": None,
                "damage_function_path": os.path.join(
                    root_dir, "../input_data/damage_function_library/CAMEL_clipped"
                ),
                "formula": "damages ~ -1 + gmsl + anomaly + np.power(anomaly, 2)",
                "subset_dict": {"ssp": ["SSP2", "SSP3", "SSP4"]},
            },
        },
    }

    os.makedirs(root_dir, exist_ok=True)
    with open(f"{root_dir}/{config_name}.yaml", "w") as file:
        yaml.dump(class_dict, file)

    return Waiter(
        path_to_config=f"{root_dir}/{config_name}.yaml", strict_combos=strict_combos
    )
