import mw_protocol.toolbox as tb
import mw_protocol.spreading as spreading
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit


# ---------------------------------- #
# ---------- MAIN METHODS ---------- #
# ---------------------------------- #

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


def remove_waterfix(ds_discharge, ds_wfix):
    """
    Remove waterfix from a discharge dataset.
    :param ds_discharge: Discharge dataset (in m3/s).#
    :param ds_wfix: Discharge dataset (in kg/m2/s).
    :return: A discharge dataset without the waterfix.
    """
    
    print("____ Removing waterfix")

    assert ds_discharge.attrs['waterfix'] is not None, "There is no waterfix on the discharge dataset."

    ds_output = ds_discharge.copy()

    wfix = ds_wfix.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    # Converting the waterfix (kg/m2/s1) to discharge unit (m3/s) and make it 3D
    wfix_fnl_3d = spreading.convert_waterfix(wfix, ds_discharge.discharge.values, 
                                            tb.surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values))
    
    ds_output.discharge.values = ds_discharge.discharge.values - wfix_fnl_3d
    ds_output.attrs['waterfix'] = None

    return ds_output
    

def add_waterfix(ds_discharge, ds_wfix, wfix_name):
    """
    Add waterfix from a discharge dataset.
    :param ds_discharge: Maps of discharge points [lat*lon].
    :return: (x indexes, y indexes, corresponding size).
    """

    print("____ Adding waterfix")

    assert ds_discharge.attrs['waterfix'] is None, "There is already a waterfix on the discharge dataset."
    
    ds_output = ds_discharge.copy()

    wfix = ds_wfix.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    # Converting the waterfix (kg/m2/s1) to discharge unit (m3/s) and make it 3D
    wfix_fnl_3d = spreading.convert_waterfix(wfix, ds_discharge.discharge.values, 
                                            tb.surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values))
    
    ds_output.discharge.values = ds_discharge.discharge.values + wfix_fnl_3d
    ds_output.attrs['waterfix'] = wfix_name

    return ds_output


def create_discharge_ts(ds_discharge, ds_lsm, rmean=None, details='low', units=None):
    """
    Create the discharge series for plot_discharge_ts.
    :param ds_discharge: Dataset with discharge to plot.
    :param ds_lsm: Dataset with land_sea_mask.
    :param rmean: running mean years
    :param details: clustering complexity, low shows all the zones.
    :param units: force writing units.
    :return: Discharge time series in Sv.
    """

    print("__ Creating discharge time series")

    lat, lon = spreading.LatAxis(ds_discharge.latitude.values), spreading.LonAxis(ds_discharge.longitude.values)
    umgrid = spreading.Grid(lat, lon)
    
    masked = np.copy(ds_lsm.lsm.values)  # land mask True (1) on land
    depthm = np.ma.masked_less(ds_lsm.depthdepth.values, 500.0)  # mask areas shallower than 500m
    masked_500m = np.copy(depthm.mask) + masked  # create binary mask from depth data
    
    collection_boxes = spreading.generate_collection_boxes()
    spread_regions = spreading.generate_spreading_regions(collection_boxes, umgrid, masked, masked_500m)

    n_lat, n_lon, n_t = len(ds_discharge.latitude), len(ds_discharge.longitude), len(ds_discharge.t)
    surface_matrix_3d = np.resize(tb.surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values), (n_t, n_lat, n_lon))

    if 'field672' in ds_discharge.keys():
         ds_discharge.rename({'field672':'discharge'})

    def to_Sv(discharge, unit):

        # Water density
        d = 1000

        if unit == 'm3/s':
            return discharge * 10**(-6)
        elif unit == 'kg/s':
            return discharge / d * 10**(-6)
        elif unit == 'kg/m2/s':
            return discharge * surface_matrix_3d / d * 10**(-6)
        elif unit == 'Sv':
            return discharge
        else:
            raise ValueError("____ Mode not recognized")

    if units is None:
        units = ds_discharge.discharge.units
    dischargeSv = to_Sv(ds_discharge.discharge.values, units)
    
    class FluxRegion:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __repr__(self):
            return f"{self.name} -> {self.value}"

    if details=='low':
        fluxes = {'elwg': FluxRegion("East Laurentide and West Greenland", [0] * n_t), 
                'gin': FluxRegion("GIN seas", [0] * n_t),
                'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                'arc': FluxRegion("Arctic", [0] * n_t),
                'so': FluxRegion("Southern Ocean", [0] * n_t),
                'pac':FluxRegion("Pacific", [0] * n_t)}
        fluxes_regions = {'elwg':['US_East_Coast', 'Gulf_of_Mexico', 'LabradorSea_BaffinBay'],
                         'gin':['Atlantic_GreenlandIceland', 'EastGreenland_Iceland', 'EastIceland', 'South_Iceland',
                          'UK_Atlantic', 'Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'arc':['N_American_Arctic', 'Greenland_Arctic', 'Eurasian_Arctic'],
                         'so':['Antarctica_RossSea', 'Antarctica_AmundsenSea', 'Antarctica_WeddellSea',
                          'Antarctica_RiiserLarsonSea', 'Antarctica_DavisSea', 'Patagonia_Atlantic', 
                          'NorthNewZealand_Pacific', 'SouthNewZealand_Pacific'],
                         'pac':['Patagonia_Pacific', 'Russia_Pacific', 'East_Pacific']}
    elif details=='medium':
        fluxes = {'nea': FluxRegion("East Laurentide and West Greenland", [0] * n_t), 
                   'egi': FluxRegion("East Greenland Iceland", [0] * n_t),
                   'nsbi': FluxRegion("Nordic Seas", [0] * n_t),
                   'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                   'larc': FluxRegion("Laurentide Arctic", [0] * n_t),
                   'garc': FluxRegion("Greenland Arctic", [0] * n_t),
                   'earc': FluxRegion("Eurasian Arctic", [0] * n_t),
                   'so': FluxRegion("Southern Ocean", [0] * n_t),
                   'pac':FluxRegion("Pacific", [0] * n_t)}
        fluxes_regions = {'nea':['US_East_Coast', 'Gulf_of_Mexico', 'LabradorSea_BaffinBay'],
                         'egi':['Atlantic_GreenlandIceland', 'EastGreenland_Iceland', 'EastIceland', 'South_Iceland'],
                         'nsbi':['UK_Atlantic', 'Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'larc':['N_American_Arctic'],
                         'garc':['Greenland_Arctic'],
                         'earc':['Eurasian_Arctic'],
                         'so':['Antarctica_RossSea', 'Antarctica_AmundsenSea', 'Antarctica_WeddellSea',
                          'Antarctica_RiiserLarsonSea', 'Antarctica_DavisSea', 'Patagonia_Atlantic', 
                          'NorthNewZealand_Pacific', 'SouthNewZealand_Pacific'],
                         'pac':['Patagonia_Pacific', 'Russia_Pacific', 'East_Pacific']}
    elif details=='high':
        fluxes = {'eus': FluxRegion("US East coast", [0] * n_t), 
                   'gom': FluxRegion("Gulf of Mexico", [0] * n_t),
                   'lsbb': FluxRegion("Labrador Sea & Baffin Bay", [0] * n_t),
                   'atgi': FluxRegion("Altantic Greenland Iceland", [0] * n_t),
                   'egi': FluxRegion("East Greenland Iceland", [0] * n_t),
                   'eic': FluxRegion("East Iceland", [0] * n_t),
                   'sic': FluxRegion("South Iceland", [0] * n_t),
                   'ukat': FluxRegion("UK Atlantic", [0] * n_t),
                   'egin': FluxRegion("Eurasian GIN", [0] * n_t),
                   'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                   'larc': FluxRegion("Laurentide Arctic", [0] * n_t),
                   'garc': FluxRegion("Greenland Arctic", [0] * n_t),
                   'earc': FluxRegion("Eurasian Arctic", [0] * n_t),
                   'atrs': FluxRegion("Antarctica Ross Sea", [0] * n_t),
                   'atas': FluxRegion("Antarctica Amundsen Sea", [0] * n_t),
                   'atws': FluxRegion("Antarctica Weddell Sea", [0] * n_t),
                   'atrl': FluxRegion("Antarctica Riiser Larson Sea", [0] * n_t),
                   'atds': FluxRegion("Antarctica Davis Sea", [0] * n_t),
                   'ptat': FluxRegion("Patagonia Atlantic", [0] * n_t),
                   'nnz': FluxRegion("North New Zealand", [0] * n_t),
                   'snz': FluxRegion("South New Zealand", [0] * n_t),
                   'ptpc': FluxRegion("Patagonia Pacific", [0] * n_t),
                   'rspc': FluxRegion("Russia Pacific", [0] * n_t),
                   'epc': FluxRegion("East Pacific", [0] * n_t)}
        fluxes_regions = {'eus':['US_East_Coast'],
                         'gom':['Gulf_of_Mexico'],
                         'lsbb':['LabradorSea_BaffinBay'],
                         'atgi':['Atlantic_GreenlandIceland'],
                         'egi':['EastGreenland_Iceland'],
                         'eic':['EastIceland'],
                         'sic':['South_Iceland'],
                         'ukat':['UK_Atlantic'],
                         'egin':['Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'larc':['N_American_Arctic'],
                         'garc':['Greenland_Arctic'],
                         'earc':['Eurasian_Arctic'],
                         'atrs':['Antarctica_RossSea'],
                         'atas':['Antarctica_AmundsenSea'],
                         'atws':['Antarctica_WeddellSea'],
                         'atrl':['Antarctica_RiiserLarsonSea'],
                         'atds':['Antarctica_DavisSea'],
                         'ptat':['Patagonia_Atlantic'],
                         'nnz':['NorthNewZealand_Pacific'],
                         'snz':['SouthNewZealand_Pacific'],
                         'ptpc':['Patagonia_Pacific'],
                         'rspc':['Russia_Pacific'],
                         'epc':['East_Pacific']}
    else:
        raise ValueError("____ Option not recognized")

    for flux_region in fluxes_regions.keys():
        spread_region_loc_3d = np.zeros((n_t, n_lat, n_lon))
        for spread_region in spread_regions:
            if spread_region['name'] in fluxes_regions[flux_region]:
                spread_region_loc_3d += np.resize(spread_region['loc'].mask, (n_t, n_lat, n_lon))
        
        fluxes[flux_region].value += np.nansum(dischargeSv * spread_region_loc_3d, axis=(1, 2))
    
    fluxes['tot'] = FluxRegion("Total", np.sum([flux.value for flux in list(fluxes.values())],axis=0))
    
    # Apply running means
    if rmean:
        for key in fluxes:
            fluxes[key].values = tb.rmean(fluxes[key].value, rmean)
    
    return fluxes

