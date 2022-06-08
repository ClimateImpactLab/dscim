from setuptools import setup, find_packages


setup(
    name="dscim",
    use_scm_version=True,
    description="Install dscim package.",
    url="https://github.com/ClimateImpactLab/dscim",
    author="Impacts Team",
    packages=find_packages(exclude=["tests"]),
    license="GNU v. 3",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # install_requires=['numpy', 'xarray', 'netCDF4', 'statsmodels',
    #                   'scipy', 'click', 'impactcommon', 'impactlab-tools',
    #                   'openest>=3', 'metacsv'],
    setup_requires=["setuptools_scm"],
    # extras_require={
    #       "test": ["pytest", "pytest-mock", "black", "flake8"],
    # },
    # entry_points={
    #       'console_scripts': [
    #             'imperics = cli:impactcalculations_cli',
    #       ]
    # },
)
