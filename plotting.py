import glac_mw.glac1d_toolbox as tb
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ---------------------------------- #
# ---------- MAIN METHODS ---------- #
# ---------------------------------- #

def plot_discharge_ts(path_discharge, unit="kg/m2/s", out="save"):
    """
    Save a discharge flux panel summary plot from a discharge dataset.
    :param path_discharge: Path of the discharge nc file to plot.
    :param unit: Unit of the discharge field of the discharge nc file.
    :param out: save or plot.
    :return:
    """
    print("__ Flux time serie algorithm")
    
    ds = xr.open_dataset(path_discharge, decode_times=False)
    t = ds.t.values
    ts = create_discharge_ts(ds, unit)
    
    figMap, ((axPac, axAtl), (axGr, axArc), (axFis, axAnt)) = plt.subplots(nrows=3, ncols=2, figsize=(26, 26))
    
    axPac.plot(t, ts[0], label="Pacific", color="blue", linestyle="-")
    axPac.set_ylabel("Meltwater flux (Sv)")
    axPac.ticklabel_format(style="sci")
    axPac.legend(loc="upper right")
    axPac.set_xlabel("Years")
    axPac.set_title("Pacific")
    
    axAtl.plot(t, ts[1][0], label="Hudson Bay", color="blue", linestyle="-")
    axAtl.plot(t, ts[1][1], label="Gulf of Mexico", color="green", linestyle="-")
    axAtl.plot(t, ts[1][2], label="Labrador sea", color="orange", linestyle="-")
    axAtl.plot(t, ts[1][3], label="New England", color="purple", linestyle="-")
    axAtl.plot(t, ts[1][4], label="Gulf of St Lawrence", color="red", linestyle="-")
    axAtl.set_ylabel("Meltwater flux (Sv)")
    axAtl.ticklabel_format(style="sci")
    axAtl.legend(loc="upper right")
    axAtl.set_xlabel("Years")
    axAtl.set_title("Atlantic")
    
    axGr.plot(t, ts[2][0], label="West Greenland", color="blue", linestyle="-")
    axGr.plot(t, ts[2][1], label="East Greenland", color="green", linestyle="-")
    axGr.plot(t, ts[2][2], label="Arctic Greenland", color="orange", linestyle="-")
    axGr.set_ylabel("Meltwater flux (Sv)")
    axGr.ticklabel_format(style="sci")
    axGr.legend(loc="upper right")
    axGr.set_xlabel("Years")
    axGr.set_title("Greenland")
    
    axArc.plot(t, ts[3][0], label="Nunavut", color="blue", linestyle="-")
    axArc.plot(t, ts[3][1], label="Arctic", color="green", linestyle="-")
    axArc.plot(t, ts[3][2], label="Greenland Arctic", color="orange", linestyle="-")
    axArc.set_ylabel("Meltwater flux (Sv)")
    axArc.ticklabel_format(style="sci")
    axArc.legend(loc="upper right")
    axArc.set_xlabel("Years")
    axArc.set_title("Arctic")
    
    axFis.plot(t, ts[4], color="blue", linestyle="-")
    axFis.set_ylabel("Meltwater flux (Sv)")
    axFis.ticklabel_format(style="sci")
    axFis.set_xlabel("Years")
    axFis.set_title("Fenoscandian Ice Sheet")
    
    axAnt.plot(t, ts[5], color="blue", linestyle="-")
    axAnt.set_ylabel("Meltwater flux (Sv)")
    axAnt.ticklabel_format(style="sci")
    axAnt.set_xlabel("Years")
    axAnt.set_title("Antarctica")
    
    sav_path = f"{path_discharge[:-3]}.fluxplot.png"
    print(f"Saving at {sav_path}")
    if out == "plot":
        figMap.show()
    else:
        figMap.savefig(sav_path)

def create_discharge_ts(ds_discharge, unit):
    """
    Create the discharge series for plot_discharge_ts.
    :param ds_discharge: Dataset with discharge to plot.
    :param unit: Unit of ds_discharge.
    :return: Discharge time series in Sv.
    """
    n_t = len(ds_discharge.t.values)
    if unit == "Sv":
        values = ds_discharge.discharge.values
    elif unit == "m3/s":
        values = ds_discharge.discharge.values * 10 ** (-6)
    elif unit == "kg/m2/s":
        ds_values = np.where(np.isnan(ds_discharge.discharge.values), 0, ds_discharge.discharge.values)
        values = np.multiply(ds_values / 1000 * 10 ** (-6),
                             tb.surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values))
    else:
        values = ds_discharge.discharge.values
    longitudes, latitudes = ds_discharge.longitude.values, ds_discharge.latitude.values
    
    lon_ant_min, lon_ant_max, lat_ant_min, lat_ant_max = 0, 359, -80, -55
    flux_ant = [0] * n_t
    lon_pac_min, lon_pac_max, lat_pac_min, lat_pac_max = 120, 250, 20, 68
    flux_pac = [0] * n_t
    lon_hud_min, lon_hud_max, lat_hud_min, lat_hud_max = 260, 290, 50, 70
    flux_hud = [0] * n_t
    lon_gm_min, lon_gm_max, lat_gm_min, lat_gm_max = 260, 280, 17, 30
    flux_gm = [0] * n_t
    lon_ls_min, lon_ls_max, lat_ls_min, lat_ls_max = 280, 315, 50, 80
    flux_ls = [0] * n_t
    lon_ne_min, lon_ne_max, lat_ne_min, lat_ne_max = 275, 315, 30, 45
    flux_ne = [0] * n_t
    lon_gsl_min, lon_gsl_max, lat_gsl_min, lat_gsl_max = 290, 315, 45, 50
    flux_gsl = [0] * n_t
    lon_wgr_min, lon_wgr_max, lat_wgr_min, lat_wgr_max = 280, 316, 58, 80
    flux_wgr = [0] * n_t
    lon_egr_min, lon_egr_max, lat_egr_min, lat_egr_max = 316, 345, 60, 80
    flux_egr = [0] * n_t
    lon_agr_min, lon_agr_max, lat_agr_min, lat_agr_max = 280, 350, 80, 85
    flux_agr = [0] * n_t
    lon_fis1_min, lon_fis1_max, lat_fis1_min, lat_fis1_max = 0, 35, 60, 80
    lon_fis2_min, lon_fis2_max, lat_fis2_min, lat_fis2_max = -20, 0, 45, 70
    flux_fis = [0] * n_t
    lon_nua_min, lon_nua_max, lat_nua_min, lat_nua_max = 200, 280, 70, 89
    flux_nua = [0] * n_t
    lon_arc_min, lon_arc_max, lat_arc_min, lat_arc_max = 0, 200, 70, 89
    flux_arc = [0] * n_t
    lon_garc_min, lon_garc_max, lat_garc_min, lat_garc_max = 280, 360, 80, 89
    flux_garc = [0] * n_t
    
    for t in range(n_t):
        flux_ant[t] = tb.sum_rect_zone(values[t], lon_ant_min, lon_ant_max, lat_ant_min, lat_ant_max, longitudes,
                                       latitudes, None)
        flux_pac[t] = tb.sum_rect_zone(values[t], lon_pac_min, lon_pac_max, lat_pac_min, lat_pac_max, longitudes,
                                       latitudes, None)
        flux_hud[t] = tb.sum_rect_zone(values[t], lon_hud_min, lon_hud_max, lat_hud_min, lat_hud_max, longitudes,
                                       latitudes, None)
        flux_gm[t] = tb.sum_rect_zone(values[t], lon_gm_min, lon_gm_max, lat_gm_min, lat_gm_max, longitudes, latitudes,
                                      None)
        flux_ls[t] = tb.sum_rect_zone(values[t], lon_ls_min, lon_ls_max, lat_ls_min, lat_ls_max, longitudes, latitudes,
                                      None)
        flux_ne[t] = tb.sum_rect_zone(values[t], lon_ne_min, lon_ne_max, lat_ne_min, lat_ne_max, longitudes, latitudes,
                                      None)
        flux_gsl[t] = tb.sum_rect_zone(values[t], lon_gsl_min, lon_gsl_max, lat_gsl_min, lat_gsl_max, longitudes,
                                       latitudes, None)
        flux_wgr[t] = tb.sum_rect_zone(values[t], lon_wgr_min, lon_wgr_max, lat_wgr_min, lat_wgr_max, longitudes,
                                       latitudes, None)
        flux_egr[t] = tb.sum_rect_zone(values[t], lon_egr_min, lon_egr_max, lat_egr_min, lat_egr_max, longitudes,
                                       latitudes, None)
        flux_agr[t] = tb.sum_rect_zone(values[t], lon_agr_min, lon_agr_max, lat_agr_min, lat_agr_max, longitudes,
                                       latitudes, None)
        flux_fis[t] = tb.sum_rect_zone(values[t], lon_fis1_min, lon_fis1_max, lat_fis1_min, lat_fis1_max, longitudes,
                                       latitudes, None) + \
                      tb.sum_rect_zone(values[t], lon_fis2_min, lon_fis2_max, lat_fis2_min, lat_fis2_max, longitudes,
                                       latitudes, None)
        flux_nua[t] = tb.sum_rect_zone(values[t], lon_nua_min, lon_nua_max, lat_nua_min, lat_nua_max, longitudes,
                                       latitudes, None)
        flux_arc[t] = tb.sum_rect_zone(values[t], lon_arc_min, lon_arc_max, lat_arc_min, lat_arc_max, longitudes,
                                       latitudes, None)
        flux_garc[t] = tb.sum_rect_zone(values[t], lon_garc_min, lon_garc_max, lat_garc_min, lat_garc_max, longitudes,
                                        latitudes, None)
        flux = flux_ant[t] + flux_pac[t] + flux_hud[t] + flux_gm[t] + flux_ls[t] + flux_ne[t] + flux_gsl[t] + flux_wgr[
            t] + flux_egr[t] + flux_agr[t] + flux_fis[t] + flux_nua[t] + flux_arc[t] + flux_garc[t]
        print(f"____ Computation time step : {t}. Total flux : {flux}")
    
    return (flux_pac, (flux_hud, flux_gm, flux_ls, flux_ne, flux_gsl), (flux_wgr, flux_egr, flux_agr),
            (flux_nua, flux_arc, flux_garc), flux_fis, flux_ant)


def scatter_mask(routed_mask):
    """
    Return parameters for a scatter plot from a routed mask.
    :param routed_mask: Maps of discharge points [lat*lon].
    :return: (x indexes, y indexes, corresponding size).
    """
    x, y, s = [], [], []
    
    for i in range(routed_mask.shape[0]):
        for j in range(routed_mask.shape[1]):
            if not np.isnan(routed_mask[i, j]) and routed_mask[i, j]:
                x.append(j), y.append(i), s.append(routed_mask[i, j])
    
    s = np.array(s) / np.max(s) * 1000
    
    return x, y, s
