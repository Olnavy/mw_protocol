import glac_mw.routing as routing
import glac_mw.spreading as spreading
import glac_mw.saving as saving
import glac_mw.plotting as plotting
import xarray as xr

experiments = ['temev', 'temfa', 'temeq', 'temel', 'temeg', 'temeb', 'temfA', 'temfv', 'temfq', 'temfl', 'temfg',
               'temfb', 'temez', 'temeu', 'temep', 'temek', 'temef', 'temea', 'temfz', 'temfu', 'temfp', 'temfk',
               'temff', 'temey', 'temet', 'temeo', 'temej',	'temee', 'temfy', 'temft', 'temfo', 'temfj', 'temfe',
               'temex',	'temes', 'temen', 'temei', 'temed', 'temfx', 'temfs', 'temfn', 'temfi', 'temfd', 'temew',
               'temer', 'temem', 'temeh', 'temec', 'temfw', 'temfr', 'temfm', 'temfh', 'temfc']

ds_hice = xr.open_dataset('/nfs/annie/eeymr/work/data/glac_mw/GLAC1DHiceF26.nc')
ds_pointer = xr.open_dataset('/nfs/annie/eeymr/work/data/glac_mw/GLAC1DdrainagePointerF26ka.nc')

for experiment in experiments:
    lsm_path = f"/nfs/annie/earpal/database/experiments/{experiment}/inidata/{experiment}.qrparm.omask.nc"
    
    ds_lsm = xr.open_dataset(lsm_path)
    
    wfix_path = f"/nfs/annie/earpal/database/experiments/{experiment}/inidata/{experiment}.qrparm.waterfix.nc"
    ds_wfix = xr.open_dataset(wfix_path)
    
    routed_mw = routing.routing(ds_hice, ds_pointer, ds_lsm, mode_flux="Volume", mode_lon="double",
                                mode_shape="cross", mode_smooth="differential")

    saving.saving(routed_mw, ds_lsm, experiment, mode="routed", start_year=-26, end_year=0, step=100,
                  mode_smooth="diff")
    
    spreaded_mw = spreading.spreading(routed_mw, ds_lsm, ds_wfix)

    saving.saving(spreaded_mw, ds_lsm, "temev", mode="spreaded", start_year=-26, end_year=0, step=100,
                  mode_smooth="diff")

plotting.plot_discharge_ts(
    "/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff_s/temev.discharge.glac_mw.nc",
    unit="kg/m2/s", out="save")
