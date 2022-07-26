import os

username = os.getenv("USER")

# path dictionaries

legacy_betas_scales = {
    "mortality": (
        "/project2/mgreenst/mortality_data/4_damage_function/damages/"
        + "mortality_damage_coefficients_quadratic_IGIA_MC_global_poly4_uclip_sharecombo_SSP3.csv",
        10**9,
    ),
    "labor": (
        f"/home/{username}/repos/labor-code-release-2020/output/ce/"
        + "damage_function_comparison/smooth_anomalies_df_mean_output_SSP3.csv",
        10**12,
    ),
}

gcm_weights = {
    "rcp45": "/project2/mgreenst/gcp/climate/SMME-weights/rcp45_2090_SMME_edited_for_April_2016.tsv",
    "rcp85": "/project2/mgreenst/gcp/climate/SMME-weights/rcp85_2090_SMME_edited_for_April_2016.tsv",
}
