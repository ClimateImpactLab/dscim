[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DSCIM: The Data-driven Spatial Climate Impact Model

This Python library enables the calculation of sector-specific partial social cost of greenhouse gases (SC-GHG) and SCGHGs that are combined across sectors using a variety of valuation methods and assumptions. The main purpose of this
library is to parse the monetized spatial damages from different sectors and integrate them
using different options ("menu options") that encompass different decisions, such as
discount levels, discount strategies, and different considerations related to
economic and climate uncertainty.

## Structure and logic

The library is split into several components that implement the hierarchy
defined by the menu options. These are the main elements of the library and
serve as the main classes to call different menu options.

```mermaid
graph TD

SubGraph1Flow(Storage and I/O)
  subgraph "Storage utilities"
  SubGraph1Flow --> A[Stacked_damages]
  SubGraph1Flow -- Climate Data --> Climate
  SubGraph1Flow -- Economic Data --> EconData
  end

  subgraph "Recipe Book"
  A[StackedDamages] --> B[MainMenu]
  B[MainMenu] --> C[AddingUpRecipe];
  B[MainMenu] --> D[RiskAversionRecipe];
  B[MainMenu] --> E[EquityRecipe]
end
```

`StackedDamages` takes care of parsing all monetized damage data from several
sectors and read the data using a `dask.distributed.Client`. At the same time,
this class takes care of ingesting FaIR GMST and GMSL data needed to draw damage
functions and calculate FaIR marginal damages to an additional emission of
carbon. The data can be read using the following components: 

Class            | Function                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Climate`        | Wrapper class to read all things climate, including GMST and GMSL. You  can pass a `fair_path` with a NetCDF with FaIR control and pulse simulations and median FaIR runs. You can use `gmst_path` to input a  CSV file with model and year anomaly data, for fitting the damage functions. |
| `EconVars`       | Class to ingest sector path related data, this includes GDP and population data. Some intermediate variables are also included in this class, check the documentation for more details                                                                                                                                                                                                             |
| `StackedDamages` | Damages wrapper class. This class contains all the elements above and  additionally reads all the computed monetized damages. A single path is needed to read all damages, and sectors must be separated by folders.  If necessary, the class will save data in `.zarr` format to make chunking operations more efficient. Check documentation of the class for more details.                      |


and these elements can be used for the menu options: 
 - `AddingUpRecipe`: Adding up all damages and collapse them to calculate a general SCC without valuing uncertainty.
 - `RiskAversionRecipe`: Add risk aversion certainty equivalent to consumption calculations - Value uncertainty over econometric and climate draws.
 - `EquityRecipe`: Add risk aversion and equity to the consumption calculations. Equity includes taking a certainty equivalent over spatial impact regions.

