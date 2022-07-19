import os, sys, yaml


def generate_configs(
    master_config,
    output,
):

    with open(master_config, "r") as stream:
        conf = yaml.safe_load(stream)

    os.makedirs(output, exist_ok=True)

    USA_sectors = {k + "_USA": v for k, v in conf["sectors"].items()}

    with open(f"{output}/AR6_ssp_global.yaml", "w") as file:
        yaml.dump(
            {
                "climate": conf["AR6_ssp_climate"],
                "sectors": conf["sectors"],
                "global_parameters": conf["global_parameters"],
                "econvars": dict(path_econ=conf["econdata"]["global_ssp"]),
            },
            file,
        )

    with open(f"{output}/AR6_ssp_USA.yaml", "w") as file:
        yaml.dump(
            {
                "climate": conf["AR6_ssp_climate"],
                "sectors": USA_sectors,
                "global_parameters": conf["global_parameters"],
                "econvars": dict(path_econ=conf["econdata"]["USA_ssp"]),
            },
            file,
        )

    with open(f"{output}/rff_global.yaml", "w") as file:
        yaml.dump(
            {
                "climate": conf["rff_climate"],
                "sectors": conf["sectors"],
                "global_parameters": conf["global_parameters"],
                "econvars": dict(
                    path_econ=f"{conf['rffdata']['socioec_output']}/rff_global_socioeconomics.nc4"
                ),
            },
            file,
        )

    with open(f"{output}/rff_USA.yaml", "w") as file:
        yaml.dump(
            {
                "climate": conf["rff_climate"],
                "sectors": USA_sectors,
                "global_parameters": conf["global_parameters"],
                "econvars": dict(
                    path_econ=f"{conf['rffdata']['socioec_output']}/rff_USA_socioeconomics.nc4"
                ),
            },
            file,
        )
