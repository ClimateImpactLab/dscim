import os
import sys
import click
import itertools
import logging
from dask.distributed import Client
from dscim import Waiter, ProWaiter


def get_logger(log_level):
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        " - ".join(["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"])
    )
    ch.setFormatter(formatter)
    logger = logging.getLogger(__file__)
    logger.setLevel(log_level)
    ch.setLevel(log_level)
    logger.addHandler(ch)

    return logger


@click.command()
@click.option(
    "--menu_order",
    default="",
    required=False,
    type=click.Choice(["damage_function", "fair", "scc", ""]),
    help="""
    Menu dish for SCC valuation. By default all menu parts will be
    executed
    """,
)
@click.option(
    "--local", is_flag=True, help="Use dask.LocalCluster() instead of a SLURMCluster()"
)
@click.option(
    "--config_file",
    "-c",
    default="",
    required=True,
    help="Path for configuration YAML file",
    type=click.Path(),
)
@click.option(
    "--pro/--not-pro",
    default=False,
    help="Whether to run in Pro mode (requires full sets of simulations, with 100s of TBs of data)",
    type=bool,
)
@click.option("--log_level", default="INFO", help="Log level. Default is INFO")
def cli(config_file, menu_order, log_level, local, pro=False):
    """Run DSCIM menu options using a configuration file

    This command-line utility will execute the menu while relying in Dask
    infrastructure and using a config file with all the user-defined
    recipe combinations and sector data. The command-line utility will run all
    the contents of the configuration file by executing the
    waiter.execute_order() function. If a menu_order is specified, then
    only the desired section will be executed.

    To use in cluster mode (no local), you have to run dask-cli to create
    workers. Within the repo you can find some examples under infrastucture
    using RCC's Midway2. Here the scheduler file will be used to be passed to
    the Dask client (by default this file is called "scheduler.json").
    """

    logger = get_logger(log_level)
    if pro:
        waiter = ProWaiter(config_file)
    else:
        waiter = Waiter(config_file)

    if local:
        client = Client()
    else:
        path_to_scheduler = os.path.join(os.getenv("SCRATCH"), "scheduler.json")
        client = Client(scheduler_file=path_to_scheduler)

    print(logger, client)

    if menu_order == "":
        waiter.execute_order()
    else:
        global_parameters = waiter.param_dict["global_parameters"]
        sectors = waiter.param_dict["sectors"].keys()

        menus = global_parameters["menu_type"]
        discounts = global_parameters["discounting_type"]
        pulse_year = waiter.param_dict["climate"]["pulse_year"]

        for menu, discount, sector in itertools.product(menus, discounts, sectors):
            obj = waiter.menu_factory(
                menu_key=menu,
                sector=sector,
                kwargs={
                    "discounting_type": discount,
                    "pulse_year": pulse_year,
                },
            )
            obj.order_plate(course=menu_order)


if __name__ == "__main__":
    cli()
