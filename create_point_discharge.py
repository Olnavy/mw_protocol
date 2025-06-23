import os
import xarray as xr
import xesmf as xe

import mw_algorithm as mw

"""
This script creates point-like freshwater fluxes from the GLAC1D ice sheet reconstruction dataset.
Please ensure that the required input datasets are available in the specified paths.
Please change the options below to suit your needs.
"""

# Inputs

## Paths
script_directory =os.getcwd()
inputs_folder = f"{script_directory}/inputs"
outputs_folder = f"{script_directory}/outputs"

## Ice sheet reconstruction
ds_ice = xr.open_dataset(f"{inputs_folder}/GLAC1DHiceF26.nc")

## Land-sea mask
ds_bathy = xr.open_dataset(f"{inputs_folder}/bathy_20ka.nc")
lsm = xr.where(ds_bathy.bathy_metry.isel(time_counter=0) > 0, 0, 1)

## River drainage map
ds_pointer = xr.open_dataset(f'{inputs_folder}/GLAC1DdrainagePointerF26ka.nc')

## Reference grid
ds_ref = xr.open_dataset(f"{inputs_folder}/ANHA4_ReNat_HydroGFD_HBC_runoff_monthly_y2003.nc", decode_times=False)


"------"

# Options
flux_unit = 'kg/s'  # Unit for freshwater flux
keep_negative = False  # Whether to keep negative values in the freshwater flux

overlapping_max_radius = 5  # Maximum radius for overlapping freshwater points

output_name = 'glac1d_freshwater_nemo_shifted.nc'  # Name of the output file

"------"

# Algorithm

## Conversion
ds_fw = mw.hi_to_discharge(ds_ice, flux_unit=flux_unit, keep_negative=keep_negative)

## Routing
IX_lon = ds_pointer.IX.sel(T40H1=-26).values*0.5 - 180.25 + 360
JY_lat = 90.125 - ds_pointer.JY.sel(T40H1=-26).values*0.25 
ds_routed = mw.routing(ds_fw, IX_lon, JY_lat)

## Regridding - To add to mw_algorithm.py
ds_fw_grid = ds_routed.rename({'lon': 'longitude', 'lat': 'latitude'})
ds_nemo_grid = ds_ref.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})

regridder = xe.Regridder(
    ds_fw_grid,
    ds_nemo_grid,
    'bilinear', 
    periodic=True,
    reuse_weights=True,
    weights=f"{outputs_folder}/regridder_glac1d_nemo.nc"
)

fw_routed_nemo = regridder(ds_routed['fw_routed'].rename({'lon': 'longitude', 'lat': 'latitude'}))

## Overlapping
ds_shifted = mw.overlapping(fw_routed_nemo, lsm, radius_max=overlapping_max_radius, verbose=False)

## Saving
ds_shifted.to_netcdf(f"{outputs_folder}/{output_name}", mode='w', format='NETCDF4_CLASSIC')
