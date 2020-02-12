import sys
sys.path.append('glac1d_meltwater')
import plotting

plotting.flux_ts("dif_test", -26, 0, True,
                 "/nfs/annie/eeymr/work/outputs/Proj_GLAC1D/dif_-26_0/teadv3.qrparm.GLAC1D_DEGLAC.nc")

