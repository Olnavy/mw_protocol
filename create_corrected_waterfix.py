import sys
sys.path.append('glac1d_meltwater')
import saving
import xarray as xr

experiments = ["teadv3", "teada3", "teaeb3", "teadb3", "teaec3", "teadc3", "teaed3", "teadd3", "teaee3", "teade3",
               "teaef3", "teadf3", "teaeg3", "teadg3", "teaeh3", "teadh3", "teaei3", "teadi3", "teaej3", "teadj3",
               "teaek3", "teadk3", "teael3", "teadl3", "teaem3", "teadm3", "teaen3", "teadn3", "teaeo3", "teado3",
               "teaep3", "teadp3", "teaeq3", "teadq3", "teaer3", "teadr3", "teaes3", "teads3", "teaet3", "teadt3",
               "teaeu3", "teadu3", "teaev3", "teadw3", "teadx3", "teady3", "teadz3", "teaea3"]

for experiment in experiments:

    print(f"Processing {experiment} waterfix")

    ds_lsm = xr.open_dataset(f"/nfs/annie/eeymr/work/data/Proj_GLAC1D/lsm/{experiment}.qrparm.omask.nc")
    ds_wfix = xr.open_dataset(f"/nfs/annie/eeymr/work/data/Proj_GLAC1D/waterfix/{experiment}.qrparam.waterfix.hadcm3.nc")

    start_date, end_date = 4000, 5000

    waterfix_patch = saving.drift_waterfix_patch("/nfs/see-fs-01_users/eeymr/dump2hold/xosfa/pg/", "xosfa", ds_wfix, start_date, end_date)
    corrected_waterfix = saving.corrected_waterfix_patch(waterfix_patch, ds_lsm, ds_wfix)

    saving.save_corrected_waterfix(ds_wfix, corrected_waterfix, experiment, start_date, end_date)