import xarray as xr
import glac_mw.routing as routing
import glac_mw.spreading as spreading
import glac_mw.saving as saving

ds_hice = xr.open_dataset('/nfs/annie/eeymr/work/data/glac_mw/GLAC1DHiceF26.nc')
ds_pointer = xr.open_dataset('/nfs/annie/eeymr/work/data/glac_mw/GLAC1DdrainagePointerF26ka.nc')
ds_wfix = xr.open_dataset("/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.waterfix.nc")
ds_lsm = xr.open_dataset("/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.omask.nc")

routed_mw = routing.routing(ds_hice, ds_pointer, ds_lsm, mode_flux="m3/s", mode_lon="double",
                            mode_shape="cross", mode_smooth="differential")

spreaded_mw = spreading.spreading(routed_mw, ds_lsm, ds_wfix)

sav_folder = "/nfs/annie/eeymr/work/outputs/glac_mw/xoup"
saving.discharge2input(spreaded_mw, ds_lsm, ds_wfix, 'xoup', 'temev', 'spreaded', -26, 0, 100,
                       "xoup.wfix.glac_ts.nc", sav_folder, mode_smooth="diff", time_shift=0)
