import glac_mw.saving as saving
import xarray as xr

ds = xr.open_dataset("/nfs/annie/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff_s/temev.wfix.glac_mw.nc",
                      decode_times=False)

ds_wfix = xr.open_dataset("/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.omask.nc",
                          decode_times=False)

saving.to_waterfix(ds, ds_wfix)
