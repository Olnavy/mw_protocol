import numpy as np
import xarray as xr
import mw_protocol.routing as routing
import mw_protocol.spreading as spreading
import mw_protocol.saving as saving
import mw_protocol.plotting as plotting

"""
Spread input ancil creation algorithm

Creates an input file by applying the routing and spreading algorithm to the GLAC-1D ice sheet reconstruction.
Please fill the inputs according to your personal folders and desired outputs.
The minimum start year is -28.000 and the maximal end year is 2.000.
"""


print("-------------------------------------")
print("---- Spread input ancil creation ----")
print("-------------------------------------\n\n")


# INPUTS PARAMETERS

script_folder = "/nfs/see-fs-01_users/eeymr/work/scripts/mw_protocol/"  # Change to your script folder
folder_name="/nfs/see-fs-01_users/eeymr/work/outputs/mw_protocol/xpfj"  # Change to your outputs folder
file_name="xpfj.wfix.glac_transient.nc"

sav_routed=True
sav_spread=True

mode_lon="double"
mode_shape="cross"
n_smooth=0
reuse_weights=False
ice_sheet='GLAC-1D'
waterfix='GLAC_1D'
flux_unit="m3/s"

start_year=None
end_year=None
time_step =None


# INPUT DATASET

print("Importing input datasets\n")

start_ky = start_year/1000 if start_year is not None else None
end_ky = end_year/1000 if end_year is not None else None

ds_lsm = xr.open_dataset(f"{script_folder}/data/temev.qrparm.omask.nc")
ds_hice = xr.open_dataset(f"{script_folder}/data/GLAC1DHiceF26.nc").sel(T122KP1=slice(start_ky, end_ky))
ds_pointer = xr.open_dataset(f"{script_folder}/data/GLAC1DdrainagePointerF26ka.nc").sel(T40H1=slice(start_ky, end_ky))
ds_wfix = xr.open_dataset(f"{script_folder}/data/temev.qrparm.waterfix.nc")


# ROUTING ALGORITHM

print("\nApplying routing algorithm\n")

ds_routed = routing.routing(ds_hice, ds_pointer, ds_lsm, flux_unit=flux_unit, mode_lon=mode_lon,
                            mode_shape=mode_shape, n_smooth=n_smooth, reuse_weights=reuse_weights,
                            ice_sheet=ice_sheet, waterfix="None")

print("Routing algorithm succeeded\n")

if sav_routed:
    print("\nSaving routed file\n")

    saving.create_output_directory(folder_name)

    if file_name[-3:]==".nc":
        ds_routed.to_netcdf(f"{folder_name}/{file_name}")
        print(f"Routed file saved at {folder_name}/routed.{file_name}")
    else:
        ds_routed.to_netcdf(f"{folder_name}/{file_name}.nc")
        print(f"Routed file saved at {folder_name}/routed.{file_name}.nc")

ds_routed.to_netcdf(f"{folder_name}/routed_test.nc")


# SPREADING ALGORITHM

print("\nApplying spreading algorithm\n")

ds_spread = spreading.spreading(ds_routed, ds_lsm, ds_wfix, discharge_unit=flux_unit, waterfix=waterfix)

print("Spreading algorithm succeeded\n")

if sav_routed:
    print("\nSaving spread file\n")

    saving.create_output_directory(folder_name)

    if file_name[-3:]==".nc":
        ds_spread.to_netcdf(f"{folder_name}/{file_name}")
        print(f"Spread file saved at {folder_name}/spread.{file_name}")
    else:
        ds_spread.to_netcdf(f"{folder_name}/{file_name}.nc")
        print(f"Spread file saved at {folder_name}/spread.{file_name}.nc")

ds_spread.to_netcdf(f"{folder_name}/spread_test.nc")


# SAVING ALGORITHM

print("\nApplying saving algorithm\n")

ds_output = saving.multiply_steps(saving.crop_years(saving.discharge_to_ancil(ds_spread), start_year, end_year), time_step)
ds_output

print("SAving algorithm succeeded\n")


# WRITING OUTPUT FILE

print("\nWriting output file\n")

saving.create_output_directory(folder_name)

if file_name[-3:]==".nc":
    ds_output.to_netcdf(f"{folder_name}/{file_name}")
    print(f"Output file saved at {folder_name}/{file_name}")
else:
    ds_output.to_netcdf(f"{folder_name}/{file_name}.nc")
    print(f"Output file saved at {folder_name}/{file_name}.nc")


print("\n\n---- End of script ----")
