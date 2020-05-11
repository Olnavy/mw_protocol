import sys
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
import xarray as xr

experiments = ["teadv3", "teada3", "teaeb3", "teadb3", "teaec3", "teadc3", "teaed3", "teadd3", "teaee3", "teade3",
               "teaef3", "teadf3", "teaeg3", "teadg3", "teaeh3", "teadh3", "teaei3", "teadi3", "teaej3", "teadj3",
               "teaek3", "teadk3", "teael3", "teadl3", "teaem3", "teadm3", "teaen3", "teadn3", "teaeo3", "teado3",
               "teaep3", "teadp3", "teaeq3", "teadq3", "teaer3", "teadr3", "teaes3", "teads3", "teaet3", "teadt3",
               "teaeu3", "teadu3", "teaev3", "teadw3", "teadx3", "teady3", "teadz3", "teaea3"]

ds_hice = xr.open_dataset("/nfs/annie/eeymr/work/data/Proj_GLAC1D/routed_fwf/GLAC1DHiceF26.nc")
ds_pointer = xr.open_dataset("/nfs/annie/eeymr/work/data/Proj_GLAC1D/routed_fwf/GLAC1DdrainagePointerF26ka.nc")

saving.create_output_folder("differential", -26, 0, True)
saving.create_output_folder("differential", -26, 0, False)

for experiment in experiments:
    lsm_path = f"/nfs/annie/eeymr/work/data/Proj_GLAC1D/lsm/{experiment}.qrparm.omask.nc"
    ds_lsm = xr.open_dataset(lsm_path)

    wfix_path = f"/nfs/annie/eeymr/work/data/Proj_GLAC1D/waterfix/{experiment}.qrparam.waterfix.hadcm3.nc"
    ds_wfix = xr.open_dataset(wfix_path)

    routed_mw = routing.routing(ds_hice, ds_pointer, ds_lsm, mode_flux="Volume", mode_lon="double",
                                mode_shape="cross", mode_smooth="differential")

    saving.saving(routed_mw, ds_lsm, experiment, mode="routed")

    spreaded_mw = spreading.spreading(routed_mw, ds_lsm, ds_wfix)

    saving.saving(spreaded_mw, ds_lsm, experiment, mode="spreaded")

plotting.flux_ts("dif", -26, 0, True,
                 f"/nfs/annie/eeymr/work/outputs/Proj_GLAC1D/dif_-26_0_s/teadv3.qrparm.waterfix_GLAC1D_DEGLAC_s.nc")
