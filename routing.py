import glac_mw.glac1d_toolbox as tb
import numpy as np


# --------------------------------- #
# ---------- MAIN METHOD ---------- #
# --------------------------------- #


def routing(ds_hice, ds_pointer, ds_lsm, mode_flux="m3/S", mode_lon="double",
            mode_shape="cross", mode_smooth="differential", t_debug=None):
    """
    Create a meltwater mass flux mask from an ice thickness reconstruction and a routing map.
    The mass flux is obtained by the formula : Fm = dh/dt*d with d=1000kg/m^3 is the density
    :param ds_hice: Ice sheet reconstruction represented by ice thickness.
    :param ds_pointer: Pointer routing map.
    :param ds_lsm: Land sea mask.
    :param mode_flux: Output flux file unit. Default is Volume.
    :param mode_lon: Longitude mode for the overlapping method (cf demo). Default is double.
    :param mode_shape: Shape mode for the overlapping method (cf demo). Default is double
    :param mode_smooth: Smoothing mode for the smoothing method (cf demo). Default is double
    :param t_debug: Number of time steps for debugging. None deactivate debugging mode. Default is None.
    :return: [t*lat*lon] numpy array. Default unit should be m3/s!
    """
    
    print("__ Routing algorithm")
    
    # Activate debuging mode.
    tmax = len(ds_hice.HGLOBH.T122KP1) if t_debug is None else t_debug
    
    # Time serie of routed files.
    land_sea_mask = ds_lsm.lsm.values
    routed_flux_serie = np.zeros(
        (tmax, land_sea_mask.shape[0], land_sea_mask.shape[1]))
    
    regridder = tb.hadcm3_regridding_method(ds_hice, ds_lsm, reuse_weights=True)
    
    for t in range(0, tmax):
        flux = hi_to_discharge(ds_hice, t, mode_flux)
        
        # Routing of the initial mask
        ix, jy = ds_pointer.IX[t].values, ds_pointer.JY[t].values
        routed_mask = routing_method(flux, ix, jy)
        
        # Regriding to HadCM3
        hadcm3_mask = regridder(routed_mask)
        hadcm3_mask = hadcm3_mask * np.sum(routed_mask) / np.sum(hadcm3_mask)  # to correct the total flux
        
        # Overlapping
        shifted_mask = overlapping_method(hadcm3_mask, land_sea_mask, mode_lon=mode_lon, mode_shape=mode_shape)
        
        # Smoothing of the results
        smoothed_mask = smoothing_method(shifted_mask, mode_smooth)
        
        # Suppressing negative values
        routed_flux_serie[t] = np.where(smoothed_mask < 0, 0, smoothed_mask)
    
    return routed_flux_serie


# ---------------------------------------- #
# ---------- ADDITIONAL METHODS ---------- #
# ---------------------------------------- #

def hi_to_discharge(ds_hice, t, flux_mode):
    """
    Convert ice thickness difference in a flux, including multiplying the flux and removing the negative values.
    :param ds_hice: Ice sheet reconstruction represented by ice thickness.
    :param t: Time step
    :param flux_mode: cf main method
    :return: [lat*lon] numpy array
    """
    
    # Water density
    d = 1000
    
    print(f"__ Computation time step : {t}.")
    
    # Computation of the flux for the time step and conversion from years to seconds.
    if t != len(ds_hice.HGLOBH.T122KP1) - 1:
        delta_t = (ds_hice.T122KP1[t + 1].values - ds_hice.T122KP1[t].values) * 365 * 24 * 3600 * 1000
        if flux_mode == "kg/s":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) * d / delta_t
        elif flux_mode == "sv":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) * 10 ** (-6) / delta_t
        elif flux_mode == "m3/s":
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) / delta_t
        else:
            print("Mode not recognized")
            flux = - (ds_hice.HGLOBH[t + 1].values - ds_hice.HGLOBH[t].values) / delta_t
    
    else:
        delta_t = (ds_hice.T122KP1[t].values - ds_hice.T122KP1[t - 1].values) * 365 * 24 * 3600 * 1000
        if flux_mode == "kg/s":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) * d / delta_t
        elif flux_mode == "sv":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) * 10 ** (-6) / delta_t
        elif flux_mode == "m3/s":
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) / delta_t
        else:
            print("Mode not recognized")
            flux = - (ds_hice.HGLOBH[t].values - ds_hice.HGLOBH[t - 1].values) / delta_t
    
    # Multiplying by the surface
    flux = np.multiply(flux, tb.surface_matrix(ds_hice.XLONGLOBP5.values, ds_hice.YLATGLOBP25.values))
    
    # Remove the negative values
    flux = np.where(flux < 0, 0, flux)
    
    return flux


def routing_method(initial_mask, ix, jy):
    """
    Create the routed mask from an initial metlwater mask and two indexes routing maps.
    :param initial_mask: Meltwater flux array [lat*lon]
    :param ix: x indexes. Correspond to a couple (lon,lat)
    :param jy: y indexes. Correspond to a couple (lon,lat)
    :return: routed mask [lat*lon]
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


def get_neighbours(radius, mode_lon, mode_shape, i_inc):
    """
    Return the (i,j) increment couples corresponding to neighbouring points for a given i_increment.
    :param radius: Radius or the research zone.
    :param mode_lon: cf demo.
    :param mode_shape: cf demo.
    :param i_inc: i_index.
    :return: (i,j) of neighbouring points.
    """
    if mode_lon == "simple" and mode_shape == "square":
        return range(-radius, radius + 1)
    elif mode_lon == "simple" and mode_shape == "cross":
        return range(-(radius - abs(i_inc)), (radius - abs(i_inc) + 1))
    elif mode_lon == "double" and mode_shape == "square":
        return range(-((radius - abs(i_inc)) // 2), (radius - abs(i_inc)) // 2 + 1)
    else:
        return range(-((radius - abs(i_inc)) // 2), (radius - abs(i_inc)) // 2 + 1)


def overlapping_method(flux_mask, lsm, mode_lon, mode_shape, verbose=False):
    """
    Shift the mask points overlapping the land mask to the closet sea point.
    :param flux_mask: Initial flux mask [lat*lon].
    :param lsm: Land sea mask.
    :param mode_lon: Simple or double longitude mode used in get_neighbours. cf demo.
    :param mode_shape: Square or cross mode used in get_neighbours. cf demo.
    :param verbose: Verbose mode.
    :return: Processed flux mask [lat*lon].
    TO DO : Very slow. How to avoid the 5* loop? Dask?
    """
    
    print(f"____ Overlapping method with {mode_lon}-{mode_shape} mode.")
    
    n_j, n_i = flux_mask.shape
    shifted_mask = np.zeros((n_j, n_i))
    
    for i in range(n_i):
        for j in range(n_j):
            
            # There is overlaping if the flux is not null and there is land.
            
            if flux_mask[j, i] != 0 and lsm[j, i] == 1:
                
                radius, land_condition, i_sea_points, j_sea_points = 0, True, [], []
                # Looping on radius while the algorithm hasn't reached sea.
                while land_condition:
                    for i_2 in range(- radius, radius + 1):
                        for j_2 in get_neighbours(radius, mode_lon, mode_shape, i_2):
                            i_test = (i + i_2) % n_i
                            j_test = min(max(j + j_2, 0), n_j - 1)
                            if lsm[j_test, i_test] == 0:  # Sea found!
                                i_sea_points.append(i_test)
                                j_sea_points.append(j_test)
                                land_condition = False
                    radius += 1
                
                if verbose:
                    print(f"____ Shifted {i, j} : {flux_mask[j, i]} -> {list(zip(i_sea_points, j_sea_points))})")
                for i_3, j_3 in zip(i_sea_points, j_sea_points):
                    shifted_mask[j_3, i_3] += flux_mask[j, i] / len(i_sea_points)
            
            # If no overlapping, the point is unchanged.
            elif not np.isnan(flux_mask[j, i]):
                shifted_mask[j, i] += flux_mask[j, i]
    
    return shifted_mask


def smoothing_method(mask, mode_smooth):
    """
    Smooth the resulting mask over time.
    :param mask: [t*lat*lon] array
    :param mode_smooth: cf demo.
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
