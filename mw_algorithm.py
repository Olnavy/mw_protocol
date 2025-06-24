import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon, box
from shapely.vectorized import contains as shply_contains
import json

import mw_toolbox as tb

# Global variables
density_ice = 917 # kg/m3


# Converstion ice sheet eleveation to flux
def hi_to_discharge(ds_ice, flux_unit='kg/s', keep_negative=False):
    """
    Convert ice elevation differences into a flux, convert the flux and remove negative values.
    :param ds_hice: ice sheet reconstruction represented by ice thickness.
    :param flux_unit: unit of the output flux. Default is kg/s.
    :param keep_negative: if True, keep negative values. Default is False.
    """
    
    def get_surface_matrix(ds_ice):
        """
        Calculate the surface area of the ice sheet.
        :param ds_ice: ice sheet reconstruction represented by ice thickness.
        :return: surface area of the ice sheet in m2.
        """
        
        ds_surface_matrix = xr.Dataset(
            {
                "surface_matrix": (["lat", "lon"],
                                    tb.calculate_surface_matrix(ds_ice.XLONGLOBP5.values, ds_ice.YLATGLOBP25.values))
            },
            coords={
                "lon": ds_ice.XLONGLOBP5.values,
                "lat": ds_ice.YLATGLOBP25.values
            }
        )

        return ds_surface_matrix
    
    def conversion(delta_elevation, delta_t, units):
        
        if units == 'm/s':
            return delta_elevation / delta_t, units
        elif units == 'm3/s':
            return delta_elevation * get_surface_matrix(ds_ice) / delta_t, units
        elif units == 'kg/m2/s':
            return delta_elevation * density_ice / delta_t, units
        elif units == 'kg/s':
            return delta_elevation * get_surface_matrix(ds_ice) * density_ice / delta_t, units
        elif units == 'Sv':
            return delta_elevation * get_surface_matrix(ds_ice) / delta_t * 1e-6, units
        else:
            print("____ The units are not yet recognised, returning m3/s")
            return delta_elevation * get_surface_matrix(ds_ice) / delta_t, 'm3/s'

    
    ds_hice = ds_ice.rename({'T122KP1': 'time', 'XLONGLOBP5': 'lon', 'YLATGLOBP25': 'lat'})
    
    # Convert ice thickness to volume flux
    delta_t = ds_ice.T122KP1.diff(dim='T122KP1')[0].values*365*24*3600*1000 # years to seconds
    delta_elevation = xr.concat(
        [xr.zeros_like(ds_hice.HGLOBH.isel(time=0)), ds_hice.HGLOBH.diff(dim='time')], dim='time')
    
    # Convert flux to discharge
    flux, units = conversion(delta_elevation, delta_t, flux_unit)
    
    # Remove negative values
    if not keep_negative:
        flux = flux.where(flux < 0, 0)

    return flux.rename({'surface_matrix': 'fw_discharge'}).transpose(
        'time', 'lat', 'lon').assign_attrs(
        {'units':units, 'description': 'Freshwater discharge from the ice sheet'})


# Routing method
def routing(ds_fw, ix, jy):
    """
    Create the routed dataset from an initial meltwater dataset and two indexes from the drainage map.
    :param initial_mask: Meltwater flux array [lat*lon]
    :param ix: x indexes. Correspond to a couple (lon,lat)
    :param jy: y indexes. Correspond to a couple (lon,lat)
    :return: routed mask [lat*lon]
    """

    print(f"____ Routing method.")

    fw_discharge = ds_fw.fw_discharge.values
    n_t, n_lat, n_lon = fw_discharge.shape
    routed_discharge = np.zeros_like(fw_discharge)

    # Create a mask for valid discharge values
    valid_mask = np.logical_and(fw_discharge != 0, ~np.isnan(fw_discharge))

    # Use numpy's indexing to route discharge values
    valid_indices = np.argwhere(valid_mask)
    for t_time, j_lat, i_lon in valid_indices:
        i_glac1d = int(2 * (ix[j_lat, i_lon] - 180.25))
        j_glac1d = int(4 * (jy[j_lat, i_lon] + 89.875))
        routed_discharge[t_time, j_glac1d, i_glac1d] += fw_discharge[t_time, j_lat, i_lon]
        
    # Create the routed dataset
    ds_routed = xr.Dataset(
        {
            "fw_routed": (["time", "lat", "lon"], routed_discharge)
        },
        coords={
            "time": ds_fw.time,
            "lon": ds_fw.lon,
            "lat": ds_fw.lat
        }
    )

    return ds_routed



# Overlapping method
def overlapping(flux_mask, lsm, radius_max=10, verbose=False):
    """
    Shift the mask points overlapping the land mask to the closet sea point.
    :param flux_mask: Initial flux mask [t*y*x].
    :param lsm: Land sea mask.
    :param radius_max: Maximum radius for the closest sea points. Default is 10.
    :param verbose: Verbose mode. Default is True.
    :return: Processed flux mask [y*x].
    """

    n_j, n_i = flux_mask[0].shape

    lsm_expanded = lsm.expand_dims({'time': flux_mask.shape[0]})
    lsm_expanded = lsm_expanded.assign_coords(time=flux_mask.time)
    overlapping_mask = np.logical_and(flux_mask!=0, lsm_expanded == 1)
    
    shifted_mask = flux_mask * ~overlapping_mask

    def sea_neighbours(i, j):
        """
        Find the closest sea neighbours for a point (j, i).
        """
        for radius in range(1, radius_max + 1):
            i_offsets, j_offsets = range(-radius, radius + 1), range(-radius, radius + 1)
            for i_offset in i_offsets:
                for j_offset in j_offsets:
                    i_test, j_test = (i + i_offset) % n_i, min(max(j + j_offset, 0), n_j - 1)
                    if lsm[j_test, i_test] == 0:
                        return [(i_test, j_test)], radius
        print(f"____ Warning: Radius max reached for ({i}, {j})")
        return [], radius_max

    for t in range(overlapping_mask.shape[0]):
        for j, i in np.argwhere(overlapping_mask[t].values):
            sea_points, radius = sea_neighbours(i, j)
            if verbose:
                print(f"____ Shifted (t={t}, {i}, {j}): {flux_mask[t, j, i]} -> {sea_points} (Radius {radius})")
            for i_sea, j_sea in sea_points:
                shifted_mask[t, j_sea, i_sea] += flux_mask[t, j, i] / len(sea_points)

    return shifted_mask



# Spreading method

def spreading(ds_fw, collection_boxes, spreading_regions, grid):
    """
        From an initial discharge, return a new dataset with meltwater collected and spread over new regions and add the
    waterfix.
    :param ds_fw: Input discharge dataset [t*y*x].
    :param collection_boxes: collection boxes dictionary.
    :param spreading_regions: sprewading regions dictionary.
    :param grid: Grid object for the model.
    :return: Spreaded discharge cube [t*y*x].
    """


    collection_masks, collection_discharges = {}, {}
    spread_masks, spread_discharges = {}, {}

    for region in spreading_regions.keys():
        print(f"Processing region: {region}")

        collection_masks[region] = np.logical_or.reduce([collection_boxes[name].get_mask() for name in spreading_regions[region]['collection_boxes']])

        collection_discharges[region] = ds_fw.fw_discharge.where(collection_masks[region]).sum(dim=('x','y'))

        spread_masks[region] = np.logical_and(
            np.logical_or.reduce([collection_boxes[name].get_mask() for name in spreading_regions[region]['spreading_boxes']]), 
            grid.get_depth_mask(depth=500))
        no_cell = np.sum(spread_masks[region].values)
        
        if no_cell != 0:
            spread_discharges[region] = xr.DataArray(
                collection_discharges[region].values[:, None, None] * spread_masks[region].values[None, :, :] / np.sum(spread_masks[region].values),
                dims=("time", "y", "x"),
                coords={"time": ds_fw.time, "y": ds_fw.y, "x": ds_fw.x},
                name="fw_discharge"
            )

    spread_discharge = sum([spread_discharges[region] for region in spread_discharges.keys()])
    return spread_discharge


class Grid:
    
    def __init__(self, lon, lat, ds_bathy):
        self.lon, self.lat = self.normalize_lon(lon), lat
        self.x, self.y = ds_bathy.x, ds_bathy.y
        self.bathy = ds_bathy.bathy_metry.isel(time_counter=0)
        self.lsm = self.set_lsm()
        
    def set_lsm(self):
        """
        Create a land-sea mask from bathymetry data.
        Land is represented by 0 and sea by 1.
        """
        lsm = xr.where(self.bathy > 0, 0, 1)
        return lsm

    def get_depth_mask(self,depth=0):
        """
        Create a mask for points below a certain depth.
        :param depth: Depth threshold. Default is 0.
        :return: Mask where points below the depth are True.
        """
        return xr.where(self.bathy > depth, True, False)

    def normalize_lon(self, lon):
        """
        Normalize longitude values to the range [0, 360).
        This is useful for consistent geographic representation.
        """
        lon_flat = np.ravel(lon)
        lon_flat = (lon_flat + 360) % 360
        return lon_flat.reshape(lon.shape)

    def polygon_mask(self, polygon):
        """
        !! Outdated
        Create a mask for points inside a polygon.
        :param polygon: Shapely Polygon object defining the area of interest.
        :return: Mask where points inside the polygon are True.
        """
        
        # Flatten the lon/lat arrays for vectorized operation
        lon_flat, lat_flat = np.ravel(self.lon), np.ravel(self.lat)

        # Use shapely.vectorized.contains for fast masking
        mask_flat = shply_contains(polygon, lon_flat, lat_flat)
        mask = mask_flat.reshape(self.lon.shape)

        return mask


class CollectionBox:
    
    def __init__(self, lonmin, lonmax, latmin, latmax, region, grid):
        self.lonmin, self.lonmax= self.normalize_lon(lonmin), self.normalize_lon(lonmax)
        self.latmin, self.latmax= latmin, latmax
        self.region = region
        self.grid= grid
        
    def normalize_lon(self,lon):
        """
        Normalize longitude values to the range [0, 360).
        This is useful for consistent geographic representation.
        """
        return (lon + 360) % 360

    
    def get_box(self):
        # return box(self.lonmin, self.latmin, self.lonmax, self.latmax)
        return Polygon(
            [(self.lonmin, self.latmin), (self.lonmax, self.latmin),
             (self.lonmax, self.latmax), (self.lonmin, self.latmax)]
        )


    def get_mask(self):
        """
        Create a mask for points inside a polygon.
        :return: Mask where points inside the polygon are True.
        """
        
        # Flatten the lon/lat arrays for vectorized operation
        lon_flat, lat_flat = np.ravel(self.grid.lon), np.ravel(self.grid.lat)

        # Use shapely.vectorized.contains for fast masking
        mask_flat = shply_contains(self.get_box(), lon_flat, lat_flat)
        mask = mask_flat.reshape(self.grid.lon.shape)

        return mask

def get_collection_boxes(input_file, grid):
    """
    Create collection boxes dictonary from an input file.
    :param input_file: Path to the json input file containing collection box definitions.
    :param grid: Grid object for the model.
    :return: Dictionary of CollectionBox objects.
    """
        
    with open(input_file, 'r') as json_file:
        boxes_data = json.load(json_file)['collection_boxes']

    collection_boxes = {key: CollectionBox(*boxes_data[key]['coords'],
                                           boxes_data[key]['region'],
                                           grid) for key in boxes_data.keys()}
    
    return collection_boxes


def group_collection_box(collection_boxes):
    """
    Group collection boxes by region.
    :param collection_boxes: Dictionary of CollectionBox objects.
    :return: Dictionary with regions as keys and list of CollectionBox objects as values.
    """
    grouped_boxes = {}
    for box in collection_boxes.values():
        if box.region not in grouped_boxes:
            grouped_boxes[box.region] = []
        grouped_boxes[box.region].append(box)
    return grouped_boxes

def get_spreading_regions(input_file):
    """
    Create spreading regions dictionary from an input file.
    :param input_file: Path to the json input file containing spreading region definitions.
    :return: Dictionary of spreading regions.
    """
    
    with open(input_file, 'r') as json_file:
        spreading_data = json.load(json_file)['spreading_regions']

    return spreading_data