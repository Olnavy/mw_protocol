import sys
sys.path.append('glac1d_meltwater')
import spreading
import saving
import plotting
import glac1d_toolbox as tb

import numpy as np
# import glac1d_meltwater.glac1d_toolbox as tb
import xarray as xr


# --------------------------------- #
# ---------- MAIN METHOD ---------- #
# --------------------------------- #


def routing(ds_hice, ds_pointer, ds_lsm, mode_flux="Volume", mode_lon="double",
            mode_shape="cross", mode_smooth="differential", t_debug=None):
    """
    From a ice thickness data set, create a time serie of routed meltwater mass flux (or volumetric flux) masks.
    The mass flux is obtained by the formula : Fm = dh/dt*d with d=1000kg/m^3 is the density
    :param ds_hice: Ice thickness file.
    :param ds_pointer: Pointer file.
    :param ds_lsm: land sea mask
    :param mode_flux: Mode to chose the flux file unit
    :param mode_lon: Longitude mode for the overlapping method (cf demo)
    :param mode_shape: Shape mode for the overlapping method (cf demo)
    :param mode_smooth: Smoothing mode for the smoothing method (cf demo)
    :param t_debug: Debug mode by limiting the number of time steps.
    :return: t*lat*lon numpy array
    """

    print("__ Routing algorithm")

    routed_flux_serie = np.zeros(
        (len(ds_hice.HGLOBH.T122KP1), ds_lsm.lsm.values.shape[0], ds_lsm.lsm.values.shape[1]))

    regridder = tb.HadCM3_regridding_method(ds_hice, ds_lsm, reuse_weights=True)

    # Debug mode
    if t_debug is None:
        tmax = routed_flux_serie.shape[0]
    else:
        tmax = t_debug

    for t in range(0, tmax):
        flux = hi_to_discharge(ds_hice, t, mode_flux)

        # Routing of the initial mask
        ix, jy = ds_pointer.IX[t].values, ds_pointer.JY[t].values
        routed_mask = routing_method(flux, ix, jy)

        # Regriding to HadCM3
        hadcm3_mask = regridder(routed_mask)
        hadcm3_mask = hadcm3_mask * np.sum(routed_mask) / np.sum(hadcm3_mask)  # to correct the total flux

        # Overlapping
        land_sea_mask = ds_lsm.lsm.values
        shifted_mask = overlapping_method(hadcm3_mask, land_sea_mask, mode_lon=mode_lon, mode_shape=mode_shape)

        # print(
        #     f"Flux check -> initial : {np.sum(flux)}, routed : {np.sum(routed_mask)},"
        #     f" regrided : {np.sum(hadcm3_mask)}, shifted {np.sum(shifted_mask)}")
        # A loger

        # Smoothing of the results
        smoothed_mask = smoothing_method(shifted_mask, mode_smooth)

        # Suppressing negative values
        routed_flux_serie[t] = np.where(smoothed_mask < 0, 0, smoothed_mask)

    return routed_flux_serie


# ---------------------------------------- #
# ---------- CONVERSION METHODS ---------- #
# ---------------------------------------- #

def hi_to_discharge(ds_hice, t, flux_mode):
    """
    Convert ice thickness difference in a flux, including multiplying the flux and removing the negative values.
    :param ds_hice: Ice thickness file
    :param t: time step
    :param flux_mode: cf main method
    :return: m3/s lat*lon numpy array
    """

    # Water density
    d = 1000

    print(f"____ Computation time step : {t}.")

    # Compute the flux for the time step and conversion years to seconds
    if t != len(ds_hice.HGLOBH.T122KP1) - 1:
        delta_t = (ds_hice.T122KP1[t + 1].values - ds_hice.T122KP1[t].values) * 365 * 24 * 3600 * 1000
        if flux_mode == "Mass":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) * d / delta_t
        elif flux_mode == "Sv":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) * 10 ** (-6) / delta_t
        elif flux_mode == "Volume":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) / delta_t
        else:
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) / delta_t

    else:
        delta_t = (ds_hice.T122KP1[t].values - ds_hice.T122KP1[t - 1].values) * 365 * 24 * 3600 * 1000
        if flux_mode == "Mass":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) * d / delta_t
        elif flux_mode == "Sv":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) * 10 ** (-6) / delta_t
        elif flux_mode == "Volume":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) / delta_t
        else:
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) / delta_t

    # Multiplying by the surface
    flux = np.multiply(flux, tb.surface_matrix(ds_hice.XLONGLOBP5.values, ds_hice.YLATGLOBP25.values))

    # Remove the negative values
    flux = np.where(flux < 0, 0, flux)

    return flux


def smoothing_method(mask, mode_smooth):
    """
    Smooth the resulting mask over time steps.
    :param mask:
    :param mode_smooth:
    :return:
    """
    processed_mask = np.zeros(mask.shape)

    if mode_smooth == "differential":
        print(f"____ Applying mask processing with {mode_smooth} mode.")
        processed_mask[0] = mask[1] * 1 / 2  # First step
        processed_mask[len(mask) - 1] = mask[len(mask) - 2] * 1 / 2  # Last step
        for i in range(1, len(mask) - 1):
            processed_mask[i] = mask[i - 1] * 1 / 2 + mask[i] * 1 / 2
    elif mode_smooth == "no_differential":
        processed_mask = mask
    else:
        print("____ Mask processing mode not recognized.")
        processed_mask = mask

    return processed_mask


# ------------------------------------- #
# ---------- ROUTING METHODS ---------- #
# ------------------------------------- #

def routing_method(initial_mask, ix, jy):
    """
    Create the routed mask from an initial mask and a pointer.
    :param initial_mask: initial mask
    :param ix:
    :param jy:
    :return: routed mask
    """

    print(f"____ Routing method.")

    ix_lon = ix * 0.5 - 180.25 + 360
    jy_lat = 90.125 - jy * 0.25

    routed_mask = np.zeros(initial_mask.shape)
    for i in range(initial_mask.shape[0]):
        for j in range(initial_mask.shape[1]):
            if not np.isnan(initial_mask[j, i]):
                i_glac1d, j_glac1d = int(2 * (ix_lon[j, i] - 180.25)), int(4 * (jy_lat[j, i] + 89.875))
                routed_mask[j_glac1d, i_glac1d] += initial_mask[j, i]

    return routed_mask


# ---------------------------------------- #
# ---------- OVERLAPING METHODS ---------- #
# ---------------------------------------- #

def get_neighbours(radius, mode_lon, mode_shape, i):
    """
    Return the indexes of the closest neighbours given one of the index, the modes and a radius.
    :param radius: radius or the research zone
    :param mode_lon: cf main method
    :param mode_shape: cf main method
    :param i: index
    :return: list of the neighbours
    """
    if mode_lon == "simple" and mode_shape == "square":
        return range(-radius, radius + 1)
    elif mode_lon == "simple" and mode_shape == "cross":
        return range(-(radius - abs(i)), (radius - abs(i) + 1))
    elif mode_lon == "double" and mode_shape == "square":
        return range(-((radius - abs(i)) // 2), (radius - abs(i)) // 2 + 1)
    else:
        return range(-((radius - abs(i)) // 2), (radius - abs(i)) // 2 + 1)


def overlapping_method(flux_mask, lsm, mode_lon="double", mode_shape="cross", verbose=False):
    """
    Shift the points dischargep points overlaping the land mask to the closet sea point.
    :param flux_mask: initial flux mask (m3/s)
    :param lsm: land sea mask
    :param mode_lon: Simple or double longitude for the closest neighbours algorithm (unregular grids)
    :param mode_shape: Saure or cross for the closest neighbours algorithm
    :param verbose: verbose mode
    :return: shifted flux mask
    TO DO : very slow. How to avoid the 5* loop? Dask?
    """

    print(f"____ Overlapping method with {mode_lon}-{mode_shape} mode.")

    n_j, n_i = flux_mask.shape
    shifted_mask = np.zeros((n_j, n_i))

    for i in range(n_i):
        for j in range(n_j):

            if flux_mask[j, i] != 0 and lsm[j, i] == 1:
                # Flux on land implies overlapping. We therefore apply the algorithm.

                radius, land_condition, i_sea_points, j_sea_points = 0, True, [], []
                while land_condition:
                    for i_2 in range(- radius, radius + 1):
                        for j_2 in get_neighbours(radius, mode_lon, mode_shape, i_2):
                            i_test = (i + i_2) % n_i
                            j_test = min(max(j + j_2, 0), n_j - 1)
                            if lsm[j_test, i_test] == 0:
                                i_sea_points.append(i_test)
                                j_sea_points.append(j_test)
                                land_condition = False
                    radius += 1

                if verbose:
                    print(f"____ Shifted {i, j} : {flux_mask[j, i]} -> {list(zip(i_sea_points, j_sea_points))})")
                for i_3, j_3 in zip(i_sea_points, j_sea_points):
                    shifted_mask[j_3, i_3] += flux_mask[j, i] / len(i_sea_points)

            elif not np.isnan(flux_mask[j, i]):
                # Else the point stay unchanged
                shifted_mask[j, i] += flux_mask[j, i]

    return shifted_mask
