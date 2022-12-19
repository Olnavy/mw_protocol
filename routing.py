import mw_protocol.toolbox as tb
import numpy as np
import pandas as pd
import xarray as xr
import datetime


# --------------------------------- #
# ---------- MAIN METHOD ---------- #
# --------------------------------- #


def routing(ds_hice, ds_pointer, ds_lsm, flux_unit="m3/S", mode_lon="double",
            mode_shape="cross", n_smooth=0, reuse_weights=True, t_debug=None,
            ice_sheet='GLAC-1D'):
    """
    Derive a meltwater flux from an ice sheet reconstruction and a routing map.
    The flux is calculated, converted, routed, regridded and smoothed depending on the options.
    We also test for overlapping points with the closest
    The format of the ice
    Create a meltwater mass flux mask from an ice thickness reconstruction and a routing map.
    The mass flux is obtained by the formula : Fm = dh/dt*d with d=1000kg/m^3 is the density

    :param ds_hice: Ice sheet reconstruction.
    :param ds_pointer: Pointer routing map.
    :param ds_lsm: Land sea mask.
    :param flux_unit: Output flux file unit. Default is m3/S.
    :param mode_lon: Simple or double longitude mode used in get_neighbours. 
    If single, longitude and latitudes cells as a equivalent. If double, 2 longitude cells correspond to 1 latitude cell.
    Use double for HadCM3. Default is double.
    :param mode_shape: Shape of the closest neighbours zone in get_neighbours method. The options are square and cross. Default is cross.
    :param n_smooth: Running mean time step. Default is 0 (no smotthing).
    :param reuse_weights: xesfm.Regridder reuse_weghts option. Default is True.
    :param t_debug: Number of time steps for debugging. None deactivate debugging mode. Default is None.

    :return: [t*lat*lon] numpy array. Default unit is be m3/s!
    """

    print("__ Routing algorithm")

    # Activate debuging mode.
    tmax = len(ds_hice.HGLOBH.T122KP1) if t_debug is None else t_debug

    # Time serie of routed files.
    land_sea_mask = ds_lsm.lsm.values
    routed_flux_serie = np.zeros(
        (tmax, land_sea_mask.shape[0], land_sea_mask.shape[1]))

    # Create the regridder
    regridder = tb.hadcm3_regridding_method(ds_hice, ds_lsm, reuse_weights=reuse_weights)

    for t in range(0, tmax):
        # Convert elevation changes to meltwater flux
        flux = hi_to_discharge(ds_hice, t, flux_unit)

        # Routing of the initial mask
        ix, jy = ds_pointer.IX[t].values, ds_pointer.JY[t].values
        drained_mask = drainage_method(flux, ix, jy)

        # Regriding to HadCM3
        hadcm3_mask = regridder(drained_mask)
        hadcm3_mask = hadcm3_mask * np.sum(drained_mask) / np.sum(hadcm3_mask)  # to correct the total flux

        # Overlapping
        shifted_mask = overlapping_method(hadcm3_mask, land_sea_mask, mode_lon=mode_lon, mode_shape=mode_shape)

        # Smoothing of the results
        smoothed_mask = smoothing_method(shifted_mask, n_smooth)

        # Suppressing negative values
        routed_flux_serie[t] = np.where(smoothed_mask < 0, 0, smoothed_mask)


    # At last, we create the xarray dataset:

    t = (ds_hice.T122KP1[:tmax]*1000).round(0).astype(int).values

    ds = xr.Dataset({'discharge': (('t', 'latitude', 'longitude'), routed_flux_serie)},
           coords={'t': t, 'latitude': ds_lsm.latitude.values, 'longitude':  ds_lsm.longitude.values})

    ds['discharge'].attrs['units'] = flux_unit
    ds['discharge'].attrs['longname'] = 'ROUTED MELTWATER DISCHARGE'

    ds['t'].attrs['long_name'] = 'time'
    ds['t'].attrs['units'] = 'years since 0000-01-01 00:00:00'
    ds['t'].attrs['calendar'] = '360_days'

    ds['longitude'].attrs['long_name'] = 'longitude'
    ds['longitude'].attrs['actual_range'] = '0., 359.'
    ds['longitude'].attrs['axis'] = 'X'
    ds['longitude'].attrs['units'] = 'degrees_east'
    ds['longitude'].attrs['modulo'] = '360'
    ds['longitude'].attrs['topology'] = 'circular'
    
    ds['latitude'].attrs['long_name'] = 'latitude'
    ds['latitude'].attrs['actual_range'] = '-89.5, 89.5'
    ds['latitude'].attrs['axis'] = 'y'
    ds['latitude'].attrs['units'] = 'degrees_north'

    ds.attrs['title'] = "ROUTED MELTWATER DISCHARGE"
    ds.attrs['start_year'] = np.min(t)
    ds.attrs['end_year'] = np.max(t)
    ds.attrs['step'] = t[1]-t[0]
    ds.attrs['mode_lon'] = mode_lon
    ds.attrs['mode_shape'] = mode_shape
    ds.attrs['running_mean_period'] = n_smooth
    ds.attrs['ice_sheet'] = ice_sheet
    ds.attrs['waterfix'] = None
    ds.attrs['lsm'] = None
    ds.attrs['history'] = "Created on " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " by Yvan Malo Romé"

    return ds


# ---------------------------------------- #
# ---------- ADDITIONAL METHODS ---------- #
# ---------------------------------------- #

def hi_to_discharge(ds_hice, t, flux_unit):
    """
    Convert ice thickness difference in a flux, convert the flux and remove negative values.
    :param ds_hice: ice sheet reconstruction represented by ice thickness.
    :param t: time step
    :param flux_unit: cf main method
    :return: [lat*lon] numpy array
    """

    print(f"__ Computation time step : {t}.")

    # Computation of the flux for the time step and conversion from years to seconds.

    def conversion(data, unit, delta):

        # Water density
        d = 1000

        if unit == 'kg/s':
            return data ** d / delta
        elif unit == 'Sv':
            return data * 10 ** (-6) / delta
        elif unit == 'm3/s':
            return data / delta
        else:
            raise ValueError("____ Mode not recognized")
            return data / delta

    # We take the difference between two consecutive time steps and divide it by the step. Years are in ky in ds_hice. 
    if t != len(ds_hice.HGLOBH.T122KP1) - 1:
        delta_t = (ds_hice.T122KP1[t + 1].values - ds_hice.T122KP1[t].values) * 365 * 24 * 3600 * 1000  # to seconds
        flux = conversion(ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values, flux_unit, delta_t)
    else:  # the last two time steps are identical.
        delta_t = (ds_hice.T122KP1[t].values - ds_hice.T122KP1[t - 1].values) * 365 * 24 * 3600 * 1000  # to seconds
        flux = conversion(ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values, flux_unit, delta_t)

    # Multiplying by the surface
    flux = np.multiply(flux, tb.surface_matrix(ds_hice.XLONGLOBP5.values, ds_hice.YLATGLOBP25.values))

    # Filter the negative values
    flux = np.where(flux < 0, 0, flux)

    return flux


def drainage_method(converted_discharge, ix, jy):
    """
    Create the drainage mask from an initial metlwater mask and two indexes routing maps.
    :param initial_mask: Meltwater flux array [lat*lon]
    :param ix: x indexes. Correspond to a couple (lon,lat)
    :param jy: y indexes. Correspond to a couple (lon,lat)
    :return: routed mask [lat*lon]
    """

    print(f"____ Routing method.")

    ix_lon = ix * 0.5 - 180.25 + 360
    jy_lat = 90.125 - jy * 0.25

    drained_mask = np.zeros(converted_discharge.shape)
    for i in range(converted_discharge.shape[0]):
        for j in range(converted_discharge.shape[1]):
            if not np.isnan(converted_discharge[j, i]):
                i_glac1d, j_glac1d = int(2 * (ix_lon[j, i] - 180.25)), int(4 * (jy_lat[j, i] + 89.875))
                drained_mask[j_glac1d, i_glac1d] += converted_discharge[j, i]

    return drained_mask


def overlapping_method(flux_mask, lsm, mode_lon='double', mode_shape='cross', verbose=False):
    """
    Shift the mask points overlapping the land mask to the closet sea point.
    :param flux_mask: Initial flux mask [lat*lon].
    :param lsm: Land sea mask.
    :param mode_lon: Simple or double longitude mode used in get_neighbours. 
    If single, longitude and latitudes cells as a equivalent. If double, 2 longitude cells correspond to 1 latitude cell.
    Use double for HadCM3. Default is double.
    :param mode_shape: Shape of the closest neighbours zone in get_neighbours method. The options are square and cross. Default is cross.
    :param verbose: Verbose mode. Default is True.
    :return: Processed flux mask [lat*lon].
    """

    print(f"____ Overlapping method with {mode_lon}-{mode_shape} mode.")

    n_j, n_i = flux_mask.shape
    overlaping_mask = np.logical_and(flux_mask != 0, lsm == 1)
    shifted_mask = flux_mask * ~overlaping_mask  # If no overlapping, the value is unchanged

    def get_neighbours(r, i_inc):
        """
        Return the (i,j) increment couples corresponding to neighbouring points for a given i_increment.
        """
        if mode_lon == "simple" and mode_shape == "square":
            return range(-r, r + 1)
        elif mode_lon == "simple" and mode_shape == "cross":
            return range(-(r - abs(i_inc)), (r - abs(i_inc) + 1))
        elif mode_lon == "double" and mode_shape == "square":
            return range(-((r - abs(i_inc)) // 2), (r - abs(i_inc)) // 2 + 1)
        else:
            return range(-((r - abs(i_inc)) // 2), (r - abs(i_inc)) // 2 + 1)

    def sea_neighbours(i, j):
        """
        For a point (j,i) , return the closest sea neighbours. The interpretation of closest depends on mode_lon and mode_shape.
        """
        radius, land_condition, i_sea_points, j_sea_points = 0, True, [], []
        # Looping on radius while the algorithm hasn't reached sea.
        while land_condition:
            for i_2 in range(- radius, radius + 1):
                for j_2 in get_neighbours(radius, i_2):
                    i_test = (i + i_2) % n_i
                    j_test = min(max(j + j_2, 0), n_j - 1)
                    if lsm[j_test, i_test] == 0:
                        i_sea_points.append(i_test)
                        j_sea_points.append(j_test)
                        land_condition = False
            radius += 1
        return i_sea_points, j_sea_points

    for j, i in np.argwhere(overlaping_mask):

        i_sea_points, j_sea_points = sea_neighbours(i, j)
    
        if verbose:
            print(f"____ Shifted {i, j} : {flux_mask[j, i]} -> {list(zip(i_sea_points, j_sea_points))})")
        for i_3, j_3 in zip(i_sea_points, j_sea_points):
            shifted_mask[j_3, i_3] += flux_mask[j, i] / len(i_sea_points)

    return shifted_mask


def smoothing_method(mask, n_smooth=0):
    """
    Apply a running mean to the data.
    :param mask: [t*lat*lon] array
    :param n_smooth: rolling window size.
    :return: Smoothed routed time series.
    """
    if n_smooth == 0:
        return mask
    else:
        return tb.rmean(mask, n_smooth, axis=0)
