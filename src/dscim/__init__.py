import logging
import sys
import os
import imp
import yaml
import xarray
import pathlib
import inspect

from datetime import datetime
from itertools import product
import pkg_resources
from dscim.menu.simple_storage import Climate, EconVars
from dscim.menu.main_recipe import MainRecipe

__author__ = "Climate Impact Lab"

import dscim.menu.baseline
import dscim.menu.risk_aversion
import dscim.menu.equity

MENU_OPTIONS = {
    "adding_up": dscim.menu.baseline.Baseline,
    "risk_aversion": dscim.menu.risk_aversion.RiskAversionRecipe,
    "equity": dscim.menu.equity.EquityRecipe,
}


# Make attributes stick
xarray.set_options(keep_attrs=True)

# courtesy of https://github.com/pydata/xarray/blob/main/xarray/__init__.py#L32
try:
    __version__ = pkg_resources.get_distribution("dscim").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10
LOG_LEVEL = int(os.environ.get("LOG_LEVEL", logging.INFO))

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)

# logger: create here to only add the handler once!
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)


class Waiter:
    """Instantiate menu class with configuration file

    The ``Waiter`` class allows the user to order any menu option for any
    number of sectors, discount types, and levels. This class will load and
    instantiate any class passed to the global parameters of the YAML file and
    will execute the ``order_plate`` function with the desired ``course`` or
    an individual object.

    The class will find for the `integration_config.yaml` by default at the
    root folder, but any YAML file can be passed to the object.

    Parameters
    ---------
    path_to_config : str
        A path to a YAML file.

    Attributes
    ----------
    param_dict : dict
        A dictionary with all parameters parsed from the YAML file

    Methods
    -------
    menu_factory
        Utility function to instantiate a class using a dictionary with class
        names and modules
    execute_order
        Execute menu order as defined in the YAML configuration file
    """

    CONFIG_FILE_DEFAULT = "integration_config.yaml"
    main_path = os.path.abspath("__file__")

    ALLOWED_SECTORS = [
        "mortality",
        "energy",
        "labor",
        "coastal",
        "agriculture",
        "combined",
    ]
    ALLOWED_DISCOUNTING_TYPES = ["euler_ramsey", "constant", "ramsey"]
    ALLOWED_PULSE_YEARS = range(2020, 2081, 5)
    ALLOWED_MENU_OPTIONS = [
        "adding_up",
        "without uncertainty",
        "risk_aversion",
        "with uncertainty",
    ]

    def __init__(self, path_to_config=None, strict_combos=False):
        self.strict_combos = strict_combos
        if path_to_config is None:
            self.path_to_config = self.CONFIG_FILE_DEFAULT
        else:
            self.path_to_config = path_to_config

        self.logger = logging.getLogger(__name__)

        # Save logs to files
        name_config = pathlib.Path(self.path_to_config).name
        log_seed = f"{datetime.today().strftime('%Y%m%d%H%M')}_{name_config}"
        fileHandler = logging.FileHandler(f"dscim_log_{log_seed}.log")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    @property
    def param_dict(self):
        """Parse and load YAML config file"""
        with open(self.path_to_config) as config_file:
            params = yaml.full_load(config_file)

        return params

    def menu_factory(self, menu_key, sector, kwargs=None):
        """Return a defined menu option with all the parameters from the YAML
        configuration file and for a desired sector
        """

        uncertainty = None

        # limit options for code release
        if self.ALLOWED_SECTORS != "any":

            assert any(
                [sector in self.ALLOWED_SECTORS]
            ), f"Sector {sector} unavailable. Please pass one of {self.ALLOWED_SECTORS}."
        assert any(
            [kwargs["discounting_type"] in self.ALLOWED_DISCOUNTING_TYPES]
        ), f"Discounting_type {kwargs['discounting_type']} unavailable. Please pass one of {self.ALLOWED_DISCOUNTING_TYPES}."
        assert any(
            [kwargs["pulse_year"] in self.ALLOWED_PULSE_YEARS]
        ), f"Pulse_year {kwargs['pulse_year']} unavailable. Please pass one of {self.ALLOWED_PULSE_YEARS}."
        assert (
            menu_key in self.ALLOWED_MENU_OPTIONS
        ), f"{menu_key} unavailable. Please pass one of {self.ALLOWED_MENU_OPTIONS}."

        # allow additional strings for set up but limit combinations of options for code release
        if self.strict_combos:
            if menu_key == "without uncertainty":
                menu_key = "adding_up"
                uncertainty = "without uncertainty"
                kwargs["fair_aggregation"] = ["median_params"]
            elif menu_key == "with uncertainty":
                # will not get here yet. menu_key unavailable
                menu_key = "risk_aversion"
                uncertainty = "with uncertainty"
                kwargs["fair_aggregation"] = ["ce"]
            else:
                raise KeyError(
                    f"{menu_key} unavailable, but should not get to this error"
                )

            if kwargs["discounting_type"] == "ramsey":
                if (uncertainty == "without uncertainty") or (uncertainty is None):
                    kwargs["discounting_type"] = "euler_ramsey"
                elif uncertainty == "with uncertainty":
                    # should not get to this point. menu_key unavailable
                    kwargs["discounting_type"] = "euler_gwr"
                else:
                    raise KeyError(
                        f"{menu_key} with ramsey discounting unavailable, but should not get to this error"
                    )

        #         import pdb; pdb.set_trace()
        global_options = self.param_dict.get("global_parameters")
        global_kwargs = {
            k: v for k, v in kwargs.items() if k in ["fair_aggregation", "eta"]
        }

        if global_kwargs is not None:
            global_options.update(**global_kwargs)

        # Load aux classes
        econ_options = self.param_dict.get("econvars")
        econ_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.getargspec(EconVars).args
        }
        if econ_kwargs is not None:
            econ_options.update(**econ_kwargs)

        climate_options = self.param_dict.get("climate")
        climate_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.getargspec(Climate).args
        }
        if climate_kwargs is not None:
            climate_options.update(**climate_kwargs)

        # Set up
        sector_config = self.param_dict["sectors"].get(sector)
        sector_config.update({k: v for k, v in global_options.items()})
        sector_config.update(
            {
                "econ_vars": EconVars(**econ_options),
                "climate_vars": Climate(**climate_options),
            }
        )
        menu_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.getargspec(MainRecipe).args
        }
        if menu_kwargs is not None:
            sector_config.update(**menu_kwargs)

        client_inst = MENU_OPTIONS[menu_key](**sector_config)

        return client_inst

    def execute_order(self):
        """Iterate through options and execute menu options"""

        global_parameters = self.param_dict["global_parameters"]
        sectors = self.param_dict["sectors"].keys()

        menus = global_parameters["menu_type"]
        discounts = global_parameters["discounting_type"]
        course = global_parameters["course"]
        pulse_year = self.param_dict["climate"]["pulse_year"]

        for menu, discount, sector in product(menus, discounts, sectors):
            obj = self.menu_factory(
                menu_key=menu,
                sector=sector,
                kwargs={
                    "discounting_type": discount,
                    "pulse_year": pulse_year,
                },
            )
            obj.order_plate(course=course)


class ProWaiter(Waiter):
    """Allows all options, but won't work without full simulation data (100s of TBs)"""

    ALLOWED_SECTORS = "any"
    ALLOWED_DISCOUNTING_TYPES = [
        "euler_ramsey",
        "euler_gwr",
        "constant",
        "constant_model_collapsed",
        "naive_ramsey",
        "naive_gwr",
        "gwr_gwr",
    ]
    ALLOWED_PULSE_YEARS = range(2020, 2081, 5)
    ALLOWED_MENU_OPTIONS = ["adding_up", "risk_aversion", "equity"]
