import xarray as xr
import glac_mw.saving as saving
import glac_mw.plotting as plotting

experiments = ['temev', 'temfa', 'temeq', 'temel', 'temeg', 'temeb', 'temfA', 'temfv', 'temfq', 'temfl', 'temfg',
               'temfb', 'temez', 'temeu', 'temep', 'temek', 'temef', 'temea', 'temfz', 'temfu', 'temfp', 'temfk',
               'temff', 'temey', 'temet', 'temeo', 'temej', 'temee', 'temfy', 'temft', 'temfo', 'temfj', 'temfe',
               'temex', 'temes', 'temen', 'temei', 'temed', 'temfx', 'temfs', 'temfn', 'temfi', 'temfd', 'temew',
               'temer', 'temem', 'temeh', 'temec', 'temfw', 'temfr', 'temfm', 'temfh', 'temfc']

for experiment in experiments:
    ds_routed = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff/{experiment}.discharge.glac_mw.nc",
        decode_times=False)
    saving.correcting(ds_routed, new_start_year=-21, new_end_year=0)
    
    ds_spreaded = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff_s/{experiment}.discharge.glac_mw.nc",
        decode_times=False)
    saving.correcting(ds_spreaded, new_start_year=-21, new_end_year=0)
    
    # to waterfix
    
    ds_wfix = xr.open_dataset("/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.waterfix.nc")
    
    ds_routed = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-21_0_100.diff/{experiment}.discharge.glac_mw.nc",
        decode_times=False)
    saving.to_waterfix(ds_routed, ds_wfix)
    
    ds_spreaded = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-21_0_100.diff_s/{experiment}.discharge.glac_mw.nc",
        decode_times=False)
    saving.to_waterfix(ds_spreaded, ds_wfix)

plotting.plot_discharge_ts(
    "/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-21_0_100.diff_s/temev.discharge.glac_mw.nc",
    "/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.omask.nc",
    unit="kg/m2/s", out="save")
