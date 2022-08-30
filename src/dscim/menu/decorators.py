import os
import logging
import functools
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


def save(name):
    """Decorator for saving output to NetCDF or CSV format

    Parameters
    ----------
    name str
        Name of file. The file will be modified by class elements

    Returns
    -------
        None
    """

    def decorator_save(func):
        @functools.wraps(func)
        def save_wrap(self, *args, **kwargs):
            out = func(self, *args, **kwargs)
            save = out

            if (self.save_path is not None) and (name in self.save_files):

                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)

                filename = f"{self.NAME}_{self.discounting_type}_eta{self.eta}_rho{self.rho}_{name}{self.filename_suffix}"
                filename_path = os.path.join(self.save_path, filename)

                if isinstance(save, xr.DataArray):

                    save = save.rename(name).to_dataset()
                    save.attrs = self.output_attrs

                    # change `None` object to str(None)
                    for att in save.attrs:
                        if save.attrs[att] is None:
                            save.attrs.update({att: "None"})

                    self.logger.info(f"Saving {filename_path}.nc4")
                    save.to_netcdf(f"{filename_path}.nc4")

                elif isinstance(save, xr.Dataset):
                    save.attrs = self.output_attrs

                    # change `None` object to str(None)
                    for att in save.attrs:
                        if save.attrs[att] is None:
                            save.attrs.update({att: "None"})

                    self.logger.info(f"Saving {filename_path}.nc4")
                    save.to_netcdf(f"{filename_path}.nc4")

                elif isinstance(save, pd.DataFrame):
                    self.logger.info(f"Saving {filename_path}.csv")
                    save.to_csv(f"{filename_path}.csv", index=False)

            return out

        return save_wrap

    return decorator_save
