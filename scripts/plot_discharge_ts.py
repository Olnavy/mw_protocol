import glac_mw.plotting as plotting

path_discharge = '/nfs/annie/eeymr/work/outputs/proj_glac_mw/glac.-28_2_100.diff_s/temev.discharge.glac_mw.nc'
path_lsm = '/nfs/annie/earpal/database/experiments/temev/inidata/temev.qrparm.omask.nc'

plotting.plot_discharge_ts(path_discharge, path_lsm, unit="kg/m2/s", out="save")
