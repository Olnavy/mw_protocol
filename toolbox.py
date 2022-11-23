import numpy as np
import numpy.ma as ma
import xarray as xr
import xesmf as xe
import pandas as pd

# ------------------------------------- #
# ---------- GENERAL METHODS ---------- #
# ------------------------------------- #


def create_coordinate_edges(coordinates):
    """
    Create bound file for regriding
    :param coordinates: 1D and regular!
    :return:

    """
    step = coordinates[1] - coordinates[0]
    return [coordinates[0] - step / 2 + i * step for i in range(len(coordinates) + 1)]


def cell_area(n_lon, lat1, lat2):
    """
    Area of a cell on a regular lon-lat grid.
    :param n_lon: number of longitude divisions
    :param lat1: bottom of the cell
    :param lat2: top of the cell
    :return:
    """
    r = 6371000
    lat1_rad, lat2_rad = 2 * np.pi * lat1 / 360, 2 * np.pi * lat2 / 360
    return 2 * np.pi * r ** 2 * np.abs(np.sin(lat1_rad) - np.sin(lat2_rad)) / n_lon


def surface_matrix(lon, lat):
    """
    Compute a matrix with all the surfaces values.
    :param lon:
    :param lat:
    :return:
    """
    n_j, n_i = len(lat), len(lon)
    lat_b = create_coordinate_edges(lat)
    surface_matrix = np.zeros((n_j, n_i))
    for i in range(n_i):
        for j in range(n_j):
            surface_matrix[j, i] = cell_area(n_i, lat_b[j], lat_b[j + 1])
    return surface_matrix


def coordinates_to_indexes(lon_target, lat_target, longitudes, latitudes):
    return np.argmin(abs(longitudes - lon_target)), np.argmin(abs(latitudes - lat_target))


def rmean(data, n, axis=0):
    """
    Running mean calculation along a single axis
    :param data: time series.
    :param n: Size of the moving window.
    :param axis: Time axis.
    :return:
    """
    try:
        return pd.Series(data).rolling(window=n, min_periods=1, center=True, axis=axis).mean().values
    except TypeError as error:
        print(error)
        print("Returning initial tab.")
        return data


def scatter_mask(routed_mask, scalling=100):
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

    s = np.array(s) / np.max(s) * scalling

    return x, y, s


# ---------------------------------------- #
# ---------- REGRIDDING METHODS ---------- #
# ---------------------------------------- #


def hadcm3_regridding_method(ds_input, ds_hadcm3, reuse_weights=False):
    """
    Conservative regridder from any input format to HadCM3. Need to update it to detect longitude/latitude fields on any
     input file.
    :param ds_input: dataset to transform to HadCM3 grid
    :param ds_hadcm3: ANY HadCM3 dataset
    :return: Regr
    """
    
    # To change to take into account different longitudes and latitudes. Here tuned to GLAC1D ice thickness.
    lon_glac1D, lat_glac1D = ds_input.XLONGLOBP5.values, ds_input.YLATGLOBP25.values
    lon_glac1D_b, lat_glac1D_b = create_coordinate_edges(lon_glac1D), create_coordinate_edges(lat_glac1D)
    
    lon_HadCM3, lat_HadCM3 = ds_hadcm3.longitude.values, ds_hadcm3.latitude.values
    lon_HadCM3_b, lat_HadCM3_b = create_coordinate_edges(lon_HadCM3), create_coordinate_edges(lat_HadCM3)
    
    ds_out = xr.Dataset(coords={'lon': (['x'], lon_HadCM3),
                                'lat': (['y'], lat_HadCM3),
                                'lon_b': (['x_b'], lon_HadCM3_b),
                                'lat_b': (['y_b'], lat_HadCM3_b)
                                })
    
    ds_in = xr.Dataset(coords={'lon': (['x'], lon_glac1D),
                               'lat': (['y'], lat_glac1D),
                               'lon_b': (['x_b'], lon_glac1D_b),
                               'lat_b': (['y_b'], lat_glac1D_b),
                               })

    return xe.Regridder(ds_in, ds_out, 'conservative', reuse_weights=reuse_weights)


def add_extra_years(routed_flux_dataset, years=2000, step=100):
    """
    Add extra years at the begining and at the end of the file
    :param routed_flux_dataset:
    :param years:
    :param step:
    :return:
    """
    n_t, n_lat, n_lon = routed_flux_dataset.shape[0], routed_flux_dataset.shape[1], routed_flux_dataset.shape[2]
    
    t_list = list(map(int, np.arange(-int(years), n_t * step + int(years), step)))
    
    processed_dataset = np.zeros((n_t + int(years / step) * 2, n_lat, n_lon))
    processed_dataset[int(years / step):int(years / step) + n_t] = routed_flux_dataset
    for t in range(int(years / step) + n_t, n_t + int(years / step) * 2):
        processed_dataset[t] = processed_dataset[int(years / step) + n_t - 1]
    return t_list, processed_dataset


# ----- ZONAL MEAN -----


def rect_zone(lon_min, lon_max, lat_min, lat_max, longitudes, latitudes):
    """
    Return the indexes fitting the longitude and latitude or a rectangular zone.

    Parameters
    ----------
    lon_min : float
        Minimal longitude. i.e longitude of the lower left corner
    lon_max : float
        Maximal longitude. i.e longitude of the upper right corner
    lat_min : float
        Minimal longitude. i.e longitude of the lower left corner
    lat_max : float
        Minimal longitude. i.e longitude of the lower left corner

    Returns
    -------
    list
        List of couple representing y and x indexes that fit the longitudes and latitudes.

    Notes
    -----
    TO DO : The case of an unique longitude or latitude is to be handled with coordinate to index.

    """
    
    target = [lon_min, lon_max, lat_min, lat_max]
    
    # # Fit the coordinates to the class extent
    # if isinstance(OcnData) or isinstance(OcnData3D):
    #     for i in range(len(coordinates)):
    #         if coordinates[i] > 180:
    #             coordinates[i] = coordinates[i] - 360
    #         elif coordinates[i] < -180:
    #             coordinates[i] = coordinates[i] + 360
    # else:
    #     for i in range(len(coordinates)):
    #         if coordinates[i] > 360:
    #             coordinates[i] = coordinates[i] - 360
    #         elif coordinates[i] < 0:
    #             coordinates[i] = coordinates[i] + 360
    
    if target[0] == target[1] and target[2] == target[3]:
        # Single longitude and latitude
        x, y = (coordinates_to_indexes(target[0], target[2], longitudes, latitudes))
        x, y = [int(x)], [int(y)]
        return list(zip(x, y))
    
    elif target[0] == target[1]:
        # Single longitude but range of latitudes
        j = int(coordinates_to_indexes(target[0], target[2], longitudes, latitudes)[0])
        i = ma.where((latitudes >= target[2]) & (latitudes <= target[3]))[1]
        return list(zip(i, [j] * len(i)))
    
    elif target[2] == target[3]:
        # Single latitude but range of longitudes
        i = int(coordinates_to_indexes(target[0], target[2], longitudes, latitudes)[1])
        j = ma.where((longitudes >= target[0]) & (longitudes <= target[1]))[0]
        return list(zip([i] * len(j), j))
    
    else:
        
        i = ma.where((longitudes >= target[0]) & (longitudes <= target[1]))
        j = ma.where((latitudes >= target[2]) & (latitudes <= target[3]))
        return list(zip(i, j))


def avg_rect_zone(values, lon_min, lon_max, lat_min, lat_max, longitudes, latitudes, lsm=None):
    zone = rect_zone(lon_min, lon_max, lat_min, lat_max, longitudes, latitudes)[0]
    avg, n = 0, 0
    
    for i in zone[0]:
        for j in zone[1]:
            if lsm is None or lsm[j, i] == 0:
                avg += values[j, i]
                n += 1
    return avg / n if n != 0 else 0


def sum_rect_zone(values, lon_min, lon_max, lat_min, lat_max, longitudes, latitudes, lsm=None):
    zone = rect_zone(lon_min, lon_max, lat_min, lat_max, longitudes, latitudes)[0]
    sum = 0
    
    for i in zone[0]:
        for j in zone[1]:
            if lsm is None or lsm[j, i] == 0:
                sum += values[j, i]
    
    return sum
