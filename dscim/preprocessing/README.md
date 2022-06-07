### Gathering sector inputs
- For all sectors excluding coastal, files should be transferred using Globus from Battuta (see directions [here](https://gitlab.com/ClimateImpactLab/Impacts/ra-manual/-/wikis/Transferring-data#transferring-using-globus)).
- for coastal, damages can be downloaded directly from the Google Cloud [here](gs://impactlab-data/gcp/outputs/coastal/). For information on how to set up Google Cloud and access Rhodium Group outputs, read [this documentation](https://gitlab.com/ClimateImpactLab/Impacts/ra-manual/-/wikis/Transferring-data#transferring-using-google-cloud-platform).

### Converting sector inputs

Currently, sector outputs require some processing before entering the integration menu. Unfortunately, each sector requires
a different sort of processing, so there is a sector-specific function inside `dscim/utils/calculate_damages.py`.

Some general guidelines for processing sectors:

1. Ensure damages are (a) **per capita** (b) **deflated** to the appropriate year's dollars (typically 2019 USD) (c) **positive**.
Regarding this last point, _bad impacts_ of climate change should be positive while _good impacts_ of climate change are
negative. We need upright damage functions.

2. Currently, coastal does _not_ require processing.

### Getting CAMEL and AMEL inputs

Due to the enormous amount of data required to compute an integrated AMEL or CAMEL SCC, there are some extra processing steps before the menu can be run.

#### AMEL
- It is recommended to execute a simple sum on each batch file for the four noncoastal sectors. These summed files can be resaved to a new folder, so that when running the menu, only 15 files need to be opened (rather than 15 * 4). This greatly increases menu execution speed. See `/project/mgreenst/AMEL_data/` for an example of the output.
- remember to replace any `np.inf` values with `np.nan` values before summing. This is especially important when adding agriculture to other sectors.

#### CAMEL
- CAMEL damages can be read as AMEL + coastal damages when executing `adding_up` menu options. However, for risk aversion, the menu is unable to compute a CE without crashing memory. To sidestep this issue, Ian Bolliger has created a notebook [here](https://gitlab.com/ClimateImpactLab/Impacts/integration/-/blob/coastaal/main/CE_calculation-2.ipynb) which computes CAMEL SCCs. The notebook is currently set up to run on his machine, but can be altered to work on RCC. These pre-calculated CEs can then be read into the menu using `self.risk_ce_cc_path`.
