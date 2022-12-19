import xarray as xr
import mw_protocol.saving as saving

# To change for your personal folder
script_folder = "/nfs/see-fs-01_users/eeymr/work/scripts/mw_protocol/"

ds_lsm = xr.open_dataset(f"{script_folder}/data/temev.qrparm.omask.copy.nc")
# del ds_lsm.attrs['lsm']
saving.add_lsm_name(ds_lsm, "GLAC-1D_21k").to_netcdf(f"{script_folder}/data/temev.qrparm.omask.nc", format='NETCDF4_CLASSIC')