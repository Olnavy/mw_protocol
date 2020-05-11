import sys
import xarray as xr
sys.path.append('glac1d_meltwater')
import routing
import spreading
import saving
import plotting
import glac1d_toolbox as tb

# import glac1d_meltwater.routing as routing
# import glac1d_meltwater.spreading as spreading
# import glac1d_meltwater.saving as saving
# import glac1d_meltwater.plottig as plotting

start, end = -21, 0

experiments = ["teadv3", "teada3", "teaeb3", "teadb3", "teaec3", "teadc3", "teaed3", "teadd3", "teaee3", "teade3",
               "teaef3", "teadf3", "teaeg3", "teadg3", "teaeh3", "teadh3", "teaei3", "teadi3", "teaej3", "teadj3",
               "teaek3", "teadk3", "teael3", "teadl3", "teaem3", "teadm3", "teaen3", "teadn3", "teaeo3", "teado3",
               "teaep3", "teadp3", "teaeq3", "teadq3", "teaer3", "teadr3", "teaes3", "teads3", "teaet3", "teadt3",
               "teaeu3", "teadu3", "teaev3", "teadw3", "teadx3", "teady3", "teadz3", "teaea3"]

saving.create_output_folder("differential", start, end, spreaded=True, corrected=False)

for experiment in experiments:

    ds_ref = xr.open_dataset(
        f"/nfs/annie/eeymr/work/outputs/Proj_GLAC1D/dif_-26_0_s/{experiment}.qrparm.waterfix_GLAC1D_DEGLAC_s.nc",
        decode_times=False)

    saving.fixing(ds_ref, "time", "dif", start, end, experiment, corrected=False)

plotting.flux_ts("dif", start, end, True,
                 f"/nfs/annie/eeymr/work/outputs/Proj_GLAC1D/dif_{start}_{end}_s/teadv3.qrparm.waterfix_GLAC1D_DEGLAC_s.nc")