import numpy as np
import xarray as xr
import mw_protocol.routing as routing
import mw_protocol.spreading as spreading
import mw_protocol.saving as saving
import mw_protocol.plotting as plotting

"""
Respread input ancil creation algorithm

Creates an input file by respreading an input file, using a different land sea mask and waterfix.
Please fill the inputs according to your personal folders and desired outputs.
The minimum start year is -28.000 and the maximal end year is 2.000.
"""


print("---------------------------------------")
print("---- Respread input ancil creation ----")
print("---------------------------------------\n\n")


# INPUTS PARAMETERS

script_folder = "/nfs/see-fs-01_users/eeymr/work/scripts/mw_protocol/"  # Change to your script folder
folder_name="/nfs/see-fs-01_users/eeymr/work/outputs/mw_protocol/xpfj"
file_name="xpfj.ice6g_glac_ts.shift.nc"

waterfix='GLAC-1D'
flux_unit='m3/s'

start_year = None
end_year = None
time_step = None

# INPUT DATASET

print("Importing input datasets\n")

# ds_final = xr.open_dataset(f"{folder_name}/xpfj.wfix.glac_test.nc", decode_times=False)
ds_ini = xr.open_dataset(f"/nfs/see-fs-01_users/eeymr/work/outputs/mw_protocol/xpfj/xpfj.wfix.ice6g_ts.shift.nc", decode_times=False).sel(t=slice(start_year,end_year))
ds_spread_ini = saving.ancil_to_discharge(ds_ini)

ds_lsm_ini = xr.open_dataset(f"{script_folder}/data/ice6g.omask.nc")  # ICE6G
ds_lsm_fnl = xr.open_dataset(f"{script_folder}/data/temev.qrparm.omask.nc")  # GLAC-1D

ds_wfix_ini = xr.open_dataset(f"{script_folder}/data/teadv3.qrparm.waterfix.nc").rename({'unspecified':'depth'})  # ICE6G
ds_wfix_fnl = xr.open_dataset(f"{script_folder}/data/temev.qrparm.waterfix.nc")  # GLAC-1D


# RESPREADING ALGORITHM

print("\nApplying respreading algorithm\n")

ds_spread_fnl = spreading.respreading(ds_spread_ini, ds_lsm_ini, ds_wfix_ini, ds_lsm_fnl, ds_wfix_fnl,
                                      discharge_unit=flux_unit, waterfix=waterfix)

print("Respreading algorithm succeeded\n")


# SAVING ALGORITHM

print("\nApplying saving algorithm\n")

ds_output = saving.multiply_steps(saving.crop_years(saving.discharge_to_ancil(ds_spread_fnl, ds_lsm_fnl), start_year, end_year), time_step)
ds_output

print("Saving algorithm succeeded\n")


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
