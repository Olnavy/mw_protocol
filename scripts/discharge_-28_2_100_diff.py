import xarray as xr
import glac_mw.saving as saving
import glac_mw.plotting as plotting

experiments = ['temev', 'temfa', 'temeq', 'temel', 'temeg', 'temeb', 'temfA', 'temfv', 'temfq', 'temfl', 'temfg',
               'temfb', 'temez', 'temeu', 'temep', 'temek', 'temef', 'temea', 'temfz', 'temfu', 'temfp', 'temfk',
               'temff', 'temey', 'temet', 'temeo', 'temej',	'temee', 'temfy', 'temft', 'temfo', 'temfj', 'temfe',
               'temex',	'temes', 'temen', 'temei', 'temed', 'temfx', 'temfs', 'temfn', 'temfi', 'temfd', 'temew',
               'temer', 'temem', 'temeh', 'temec', 'temfw', 'temfr', 'temfm', 'temfh', 'temfc']

for experiment in experiments:

    ds_routed = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff/temev.discharge.glac_mw.nc",
        decode_times=False)

    saving.correcting(ds_routed, new_start_year=-28, new_end_year=2)


    ds_spreaded = xr.open_dataset(
        f"/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-26_0_100.diff_s/temev.discharge.glac_mw.nc",
        decode_times=False)

    saving.correcting(ds_spreaded, new_start_year=-28, new_end_year=2)

plotting.plot_discharge_ts(
    "/nfs/see-fs-01_users/eeymr/work/outputs/proj_glac_mw/glac.-28_2_100.diff_s/temev.discharge.glac_mw.nc",
    unit="kg/m2/s", out="save")
