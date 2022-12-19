import mw_protocol.toolbox as tb
import numpy as np
import xarray as xr
import datetime

"""
DIRECTLY ADAPTED FROM produce_deglacHadCM3_spread.py by R F Ivanovic
"""


# --------------------------------- #
# ---------- MAIN METHOD ---------- #
# --------------------------------- #


def spreading(ds_discharge, ds_lsm, ds_wf, discharge_unit='m3/s', waterfix='GLAC-1D', debug=False):
    """
    From an initial discharge, return a new dataset with meltwater collected and spread over new regions and add the
    waterfix.
    :param ds_discharge: Input discharge dataset [t*lat*lon].
    :param ds_lsm: Land sea mask.
    :param ds_wf: Waterfix.
    :param discharge_unit: Unit of the input discharge. default is m3/s.
    :return: Discharge cube with the waterfix [t*lat*lon].
    """
    
    print("__ Spreading algorithm")
    
    def to_m3s(data, unit):

        # Water density
        d = 1000

        if unit == 'm3/s':
            return data
        elif unit == 'kg/s':
            return data / d
        elif unit == 'Sv':
            return data * 10 ** (6)
        else:
            raise ValueError("____ Mode not recognized")

    # Discharge
    discharge = to_m3s(ds_discharge.discharge.values, discharge_unit)   # convert discharge to m3/s

    # Land sea mask
    lon_lsm, lat_lsm, depth, lsm = \
        ds_lsm.longitude.values, ds_lsm.latitude.values, ds_lsm.depthdepth.values, ds_lsm.lsm.values
    
    # Waterfix
    wfix = ds_wf.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    # Test that the three files use the same coordinate system
    assert np.array_equal(ds_wf.isel(longitude=slice(None,-2)).longitude.values, lon_lsm)
    # Converting the waterfix (kg/m2/s1) to discharge unit (m3/s) and make it 3D
    wfix_fnl_3d = convert_waterfix(wfix, discharge, tb.surface_matrix(lon_lsm, lat_lsm))
    
    # Coordinate system
    lat, lon = LatAxis(lat_lsm), LonAxis(lon_lsm)
    umgrid = Grid(lat, lon)

    # Land sea mask and surface matrix
    masked = np.copy(lsm)  # land mask True (1) on land
    depthm = np.ma.masked_less(depth, 500.0)  # mask areas shallower than 500m
    masked_500m = np.copy(depthm.mask) + masked  # create binary mask from depth data
    surface_matrix = tb.surface_matrix(lon_lsm, lat_lsm)
    
    # Step 1 : Generate collection and spreading zones
    collection_boxes = generate_collection_boxes()
    spread_regions = generate_spreading_regions(collection_boxes, umgrid, masked, masked_500m)
    
    # Step 2 : Spread the collected freshwater in the spreading zones
    spread_mw = spreading_method(discharge, spread_regions, surface_matrix)
    
    # Step 3 : Add waterfix to the spread mask
    total_mw = spread_mw + wfix_fnl_3d
    
    # Step 4 : Calculate the loss and check the al 
    others_mw = get_discharge_others(discharge, spread_regions)
    flux_check(discharge, spread_mw, total_mw, others_mw, wfix_fnl_3d)

    # We copy the input dataset and replace the discharge flux.
    ds = ds_discharge.copy()

    ds['discharge'].values = total_mw
    ds['discharge'].attrs['units'] = 'm3/s'
    ds['discharge'].attrs['longname'] = 'SPREAD MELTWATER DISCHARGE'

    ds.attrs['title'] = "SPREAD MELTWATER DISCHARGE"
    ds.attrs['waterfix'] = waterfix
    ds.attrs['history'] = f"Created {datetime.datetime.now()} by Yvan Romé"

    if debug:
        return ds, (discharge, spread_mw, total_mw, others_mw, wfix_fnl_3d)
    else:
        return ds

def spreading_method(init_mw, spread_regions, surface_matrix):
    """
    Spreading meltwater over spreading zones.
    :param init_mw: Initial discharge cube [t*lat*lon] (m3/s).
    :param spread_regions: List of spreading zones objects.
    :param surface_matrix: LSM surface matrix [lat*lon]
    :return: spread discharge cube [t*lat*lon] (m3/s).
    """
    nt, nlat, nlon = init_mw.shape
    spread_mw = np.zeros((nt, nlat, nlon))
    
    for spread_region in spread_regions:
        print(f"____ Spreading in {spread_region['name']}")
        
        spread_region_loc_3d = np.resize(spread_region['loc'].mask,
                                         (nt, nlat, nlon))  # ice melt (e.g. river) mouth location (mask)
        spread_region_discharge_ts = np.sum(init_mw * spread_region_loc_3d,
                                            axis=(1, 2))  # sum over the coordinates to create a time serie
        spread_region_discharge_3d = np.rollaxis(np.resize(spread_region_discharge_ts, (nlat, nlon, nt)),
                                                 2)  # expand time series to nt,nlat,nlon
        
        # Spread discharge to normal region to spread over (mask)
        surface_matrix_3d = np.resize(surface_matrix, (nt, nlat, nlon))
        spread_region_area_3d = np.resize(spread_region['region'].mask, (nt, nlat, nlon))
        spread_region_flux_3d = spread_region_discharge_3d / spread_region['region'].totalarea * surface_matrix_3d
        # flux spread over region
        spread_region_flux_spread_3d = np.where(spread_region_area_3d, spread_region_flux_3d, 0)
        
        # Add spread_region discharge to global field
        spread_mw = spread_mw + spread_region_flux_spread_3d
    
    return spread_mw


# ---------------------------------------- #
# ---------- RESPREADING METHOD ---------- #
# ---------------------------------------- #

def respreading(ds_discharge, ds_lsm_init, ds_wfix_init, ds_lsm_fnl, ds_wfix_fnl, discharge_unit='m3/s', waterfix='GLAC-1D', debug=False):
    """
    From an initial discharge, return a new dataset with meltwater collected and spread over new regions and add the
    waterfix.
    :param ds_discharge: Input discharge dataset [t*lat*lon].
    :param ds_lsm: Land sea mask.
    :param ds_wf: Waterfix.
    :param discharge_unit: Unit of the input discharge. default is m3/s.
    :return: Discharge cube with the waterfix [t*lat*lon].
    """
    
    print("__ Respreading algorithm")
    
    def to_m3s(data, unit):

        # Water density
        d = 1000

        if unit == 'm3/s':
            return data
        elif unit == 'kg/s':
            return data / d
        elif unit == 'Sv':
            return data * 10 ** (6)
        else:
            raise ValueError("____ Mode not recognized")

    # Discharge
    discharge = to_m3s(ds_discharge.discharge.values, discharge_unit)   # convert discharge to m3/s

    # Land sea mask
    lon_lsm, lat_lsm = ds_lsm_init.longitude.values, ds_lsm_init.latitude.values
    depth_init, depth_fnl = ds_lsm_init.depthdepth.values, ds_lsm_fnl.depthdepth.values

    lsm_init, lsm_fnl = ds_lsm_init.lsm.values, ds_lsm_fnl.lsm.values

    # Waterfix

    wfix_init = ds_wfix_init.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    wfix_fnl = ds_wfix_fnl.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    # Test that the three files use the same coordinate system
    assert np.array_equal(ds_wfix_init.isel(longitude=slice(None,-2)).longitude.values, lon_lsm)
    assert np.array_equal(ds_wfix_fnl.isel(longitude=slice(None,-2)).longitude.values, lon_lsm)
    # Converting the waterfix (kg/m2/s1) to discharge unit (m3/s) and make it 3D
    wfix_init_3d = convert_waterfix(wfix_init, discharge, tb.surface_matrix(lon_lsm, lat_lsm))
    wfix_fnl_3d = convert_waterfix(wfix_fnl, discharge, tb.surface_matrix(lon_lsm, lat_lsm))
    
    # Coordinate system
    lat, lon = LatAxis(lat_lsm), LonAxis(lon_lsm)
    umgrid = Grid(lat, lon)

    # Land sea mask and surface matrix
    masked_init = np.copy(lsm_init)  # land mask True (1) on land
    depthm_init = np.ma.masked_less(depth_init, 500.0)  # mask areas shallower than 500m
    masked_500m_init = np.copy(depthm_init.mask) + masked_init  # create binary mask from depth data

    masked_fnl = np.copy(lsm_fnl)  # land mask True (1) on land
    depthm_fnl = np.ma.masked_less(depth_fnl, 500.0)  # mask areas shallower than 500m
    masked_500m_fnl = np.copy(depthm_fnl.mask) + masked_fnl  # create binary mask from depth data

    surface_matrix = tb.surface_matrix(lon_lsm, lat_lsm)
    
    # Step 0 : Remove initial waterfix
    discharge_nowfix = discharge - wfix_init_3d

    # Step 1 : Generate collection and spreading zones
    collection_boxes = generate_collection_boxes()

    spread_regions_init = generate_spreading_regions(collection_boxes, umgrid, masked_init, masked_500m_init)
    spread_regions_fnl = generate_spreading_regions(collection_boxes, umgrid, masked_fnl, masked_500m_fnl)

    # Step 2 : Spread the collected freshwater in the spreading zones
    spread_mw = respreading_method(discharge_nowfix, spread_regions_init, spread_regions_fnl, surface_matrix)
    
    # Step 3 : Add waterfix to the spread mask
    total_mw = spread_mw + wfix_fnl_3d
    
    # Step 4 : Calculate the loss and check the 
    others_mw = get_discharge_others(spread_mw, spread_regions_fnl)
    flux_check(discharge_nowfix, spread_mw, total_mw, others_mw, wfix_fnl_3d, wfix_init_3d)
    # We copy the input dataset and replace the discharge flux.
    ds = ds_discharge.copy()

    ds['discharge'].values = total_mw
    ds['discharge'].attrs['units'] = 'm3/s'
    ds['discharge'].attrs['longname'] = 'RESPREAD MELTWATER DISCHARGE'

    ds.attrs['title'] = "RESPREAD MELTWATER DISCHARGE"
    ds.attrs['waterfix'] = waterfix
    ds.attrs['history'] = f"Created {datetime.datetime.now()} by Yvan Romé"
 
    if debug:
        return ds, (discharge_nowfix, spread_mw, total_mw, wfix_fnl_3d, wfix_init_3d)
    else:
        return ds

def respreading_method(init_mw, spread_regions_init, spread_regions_fnl, surface_matrix):
    """
    Spreading meltwater over spreading zones.
    :param init_mw: Initial discharge cube [t*lat*lon] (m3/s).
    :param spread_regions: List of spreading zones objects.
    :param surface_matrix: LSM surface matrix [lat*lon]
    :return: spread discharge cube [t*lat*lon] (m3/s).
    """
    nt, nlat, nlon = init_mw.shape
    spread_mw = np.zeros((nt, nlat, nlon))
    
    # We need to loop by regions name to get the corresponding regions in the inital and final regions.
    
    for i in range(len(spread_regions_init)):

        spread_region_init = spread_regions_init[i]
        spread_region_fnl = spread_regions_fnl[i]

        assert spread_region_init['name'] == spread_region_fnl['name'] # Are the initial and final regions the same?

        print(f"____ Spreading in {spread_region_init['name']}")
        
        spread_region_loc_3d = np.resize(spread_region_init['loc'].mask,
                                         (nt, nlat, nlon))  # ice melt (e.g. river) mouth location (mask)
        spread_region_discharge_ts = np.sum(init_mw * spread_region_loc_3d,
                                            axis=(1, 2))  # sum over the coordinates to create a time serie
        spread_region_discharge_3d = np.rollaxis(np.resize(spread_region_discharge_ts, (nlat, nlon, nt)),
                                                 2)  # expand time series to nt,nlat,nlon
        
        # Spread discharge to normal region region to spread over (mask)
        surface_matrix_3d = np.resize(surface_matrix, (nt, nlat, nlon))
        spread_region_area_3d = np.resize(spread_region_fnl['region'].mask, (nt, nlat, nlon))
        spread_region_flux_3d = spread_region_discharge_3d / spread_region_fnl['region'].totalarea * surface_matrix_3d
        # flux spread over region
        spread_region_flux_spread_3d = np.where(spread_region_area_3d, spread_region_flux_3d, 0)
        
        # Add spread_region discharge to global field
        spread_mw = spread_mw + spread_region_flux_spread_3d
    
    return spread_mw

# ----------------------------------------- #
# ---------- ADDITIONNAL METHODS ---------- #
# ----------------------------------------- #


def convert_waterfix(wfix, init_mw, surface_matrix):
    """
    Convert the waterfix to m3/S and resize it to fit the waterfix.
    :param wfix: Waterfix.
    :param init_mw: Discharge file to extract the shape from.
    :param surface_matrix: LSM surface matrix.
    :return: 3D converted waterfix [m3/S, t*lat*lon]
    """
    nt, nlat, nlon = init_mw.shape
    d = 1000  # water density
    
    wfix = np.where(np.isnan(wfix), 0, wfix)
    wfix_fnl_flux = wfix / d * surface_matrix
    wfix_fnl_3d = np.resize(wfix_fnl_flux, (nt, nlat, nlon))  # expand the 2D waterfix with time
    
    return wfix_fnl_3d


def get_discharge_others(init_mw, spread_regions):
    """
    Calculate the flux from areas off the spreading regions.
    :param init_mw: Flux array [t*lat*lon]
    :param spread_regions: List of spreading zones objects.
    :return: Flux cube [t*lat*lon] of remaining flux after the spreading algorithm.
    """
    discharge_others = np.ma.copy(init_mw)
    nt, nlat, nlon = init_mw.shape
    
    # Get all the regions covered by the spread_regions mask
    all_regions_loc = spread_regions[0]['loc'].mask[:]
    for spread_region in spread_regions:
        all_regions_loc = np.logical_or(spread_region['loc'].mask, all_regions_loc)
    all_regions_loc_3d = np.resize(all_regions_loc, (nt, nlat, nlon))
    
    discharge_others[all_regions_loc_3d] = 0  # set discharge to 0 within spread regions
    
    return discharge_others


def flux_check(init_mw, spread_mw, total_mw, others_mw, wfix_fnl_3d, wfix_init_3d=None):
    """
    Quick display of flux check. The flux displayed is the temporal mean. 0.1% error.
    :param init_mw: Initial discharge cube, without waterfix!! [t*lat*lon].
    :param spread_mw: Sreaded discharge cube [t*lat*lon].
    :param others_mw: leftovers discharge cube [t*lat*lon].
    :param wfix_fnl_3d: Waterfix [t*lat*lon].
    :param total_mw: Initial discharge cube [t*lat*lon].
    :return: None
    """
    
    if wfix_init_3d is None:
        wfix_init_3d = wfix_fnl_3d

    # Check the spreading has worked
    nt = init_mw.shape[0]
    
    init_flux = np.sum(init_mw) / float(nt)
    wfix_init_flux = np.sum(wfix_init_3d) / float(nt)
    total_init_flux = np.sum(init_mw + wfix_init_3d) / float(nt)
    # total_flux_init = init_flux + wfix_init_flux

    spread_flux = np.sum(spread_mw) / float(nt)
    wfix_fnl_flux = np.sum(wfix_fnl_3d) / float(nt)
    others_flux = np.sum(others_mw) / float(nt)
    total_fnl_flux = np.sum(total_mw) / float(nt)

    try:
        assert abs(init_flux - (spread_flux + others_flux))<= init_flux * 10 ** (-3)
        print("Spreading of water succeded")
    except AssertionError as error:
        print(error)
        print("Spreading of water didn't work.")
    
    # Calculate stats on output field
    print("\nChecking the mean flux (from m3/s to Sv, 0.1% percent error).")
    
    print(f'[1] Initial flux (Sv): {init_flux*10**(-6):.4f}')
    print(f'[2] Initial waterfix (Sv): {wfix_init_flux*10**(-6):.4f}')
    print(f'[3] Initial flux + waterfix (Sv): {total_init_flux*10**(-6):.4f}')

    print(f'[4] Spread flux (Sv): {spread_flux*10**(-6):.4f}')
    print(f'[5] Other zones flux (Sv): {others_flux*10**(-6):.4f}')
    print(f'[6] Final waterfix (Sv): {wfix_fnl_flux*10**(-6):.4f}')
    print(f'[7] Final flux  (Sv): {total_fnl_flux*10**(-6):.4f} \n')
    
    print('Initial fluxes : [3] = [1] + [2]:',
          abs(total_init_flux - (init_flux + wfix_init_flux)) <= (total_init_flux * 10 ** (-1)))
    
    print('Final fluxes : [7] = [4] + [5] + [6]:',
          abs(total_fnl_flux - (spread_flux + others_flux + wfix_fnl_flux)) <= (total_fnl_flux * 10 ** (-1)))

    print('Success of the algorithm : [1] = [4] + [5]:',
          abs(init_flux - (spread_flux + others_flux)) <= (init_flux * 10 ** (-1)))


# ------------------------------------------------------------ #
# ---------- COLLECTION BOXES AND SPREADING REGIONS ---------- #
# ------------------------------------------------------------ #


def generate_collection_boxes():
    """
    Method to define the collection boxes.
    :return: Dictionary of collection Boxes.
    """
    
    collection_boxes = dict()
    # Define rivers (automatically defines masks and areas)
    # Start with defining the new regions to put the water
    
    # USA East Coast
    collection_boxes["USECoast1"] = Box(37, 46, -70, -52)
    collection_boxes["USECoast2"] = Box(32, 37, -70, -65)
    collection_boxes["USECoast3"] = Box(28.75, 43, -81, -70)
    collection_boxes["USECoast4"] = Box(40, 46, -52, -48)
    collection_boxes["USECoast5"] = Box(46, 50, -66, -58)
    collection_boxes["USECoast6"] = Box(40, 46, -48, -46)  # New One, only for catching

    # Greenland Arctic
    collection_boxes["GrArc1"] = Box(81, 88, 279.5, 346)
    # North American Arctic
    collection_boxes["NAMArc1"] = Box(78, 86, 271, 279.5)
    collection_boxes["NAMArc2"] = Box(68.75, 86, 246, 271)
    collection_boxes["NAMArc3"] = Box(60, 82, 233, 246)
    collection_boxes["NAMArc4"] = Box(60, 80, 191, 233)
    collection_boxes["NAMArc5"] = Box(55, 68.75, 250, 264.375)  # only for catching the water, not for spreading it
    collection_boxes["NWTerr1"] = Box(55, 60, 235, 246)  # only for catching the water
    collection_boxes["NWTerr2"] = Box(55, 66, 246, 250)  # not for spreading it
    # Great Lakes  # Can decide which spreading box to add this to
    collection_boxes["GrLakes1"] = Box(43, 48.75, -90, -72)  # only for catching the water, not for spreading it
    # Gulf of Mexico
    collection_boxes["GoM1"] = Box(17.7, 28.75, -96.3, -80)
    # East Pacific
    collection_boxes["EPac1"] = Box(50, 60, 191, 215.5)
    collection_boxes["EPac2"] = Box(50, 60, 215.5, 225.5)
    collection_boxes["EPac3"] = Box(38.5, 60, 225.5, 234.5)
    collection_boxes["EPac4"] = Box(33.75, 38.5, 230, 260)
    collection_boxes["EPac5"] = Box(28.5, 33.75, 234.5, 260)
    # Russia Pacific
    collection_boxes["RussPac1"] = Box(58, 68, 178, 191)
    # Labrador Sea & Baffin Bay
    collection_boxes["BafLab1"] = Box(68.75, 80, 275, 317)
    collection_boxes["BafLab2"] = Box(50, 68.75, 294.25, 317)
    collection_boxes["BafLab3"] = Box(46, 50, 305.75, 317)
    collection_boxes["HudBay1"] = Box(48.75, 68.75, 264.375, 294.375)  # only for catching the water
    collection_boxes["HudBay2"] = Box(51, 54, 260, 264.375)  # not for spreading it
    # Atlantic Greenland Iceland
    collection_boxes["AtlGr1"] = Box(58, 71.25, 317, 337.25)
    collection_boxes["AtlGr2"] = Box(62.5, 63.75, 337.25, 339.5)
    # E Greenland & Iceland
    collection_boxes["EGrIce1"] = Box(63.75, 81, 337.25, 346)
    collection_boxes["EGrIce2"] = Box(68.75, 83, 346, 357)
    # E Iceland
    collection_boxes["EIceland1"] = Box(63.75, 68.75, 346, 351)
    # UK Atlantic
    collection_boxes["UKAtl1"] = Box(46, 62.5, 346.75, 360)
    # Eurasian GIN Seas
    collection_boxes["EurGIN1"] = Box(68, 80, 3, 9.5)
    collection_boxes["EurGIN2"] = Box(68, 78, 9.5, 24.375)
    collection_boxes["EurGIN3"] = Box(60, 68, 0, 16)
    collection_boxes["EurGIN4"] = Box(50, 60, 0, 13)
    collection_boxes["EurGIN5"] = Box(66.25, 68, 16, 24.375)
    collection_boxes["EurGIN6"] = Box(68, 80, 0., 3)  # New one, only for catching
    collection_boxes["Baltic1"] = Box(50, 60.0, 13, 30)  # only for catching the water
    collection_boxes["Baltic2"] = Box(60, 66.25, 16, 38)  # not for spreading
    # South Iceland
    collection_boxes["SIceland1"] = Box(60, 63.75, 339.5, 346.75)
    # Siberian Arctic
    collection_boxes["SibArc1"] = Box(68, 82, 173, 191)
    collection_boxes["SibArc2"] = Box(68, 82, 114.5, 173)  # New One
    # Eurasian Arctic
    collection_boxes["EurArc1"] = Box(78, 86, 9.5, 114.5)
    collection_boxes["EurArc2"] = Box(66.25, 78, 24.375, 114.5)
    collection_boxes["EurArc3"] = Box(80, 86, 0, 9)  # New One - only for catching
    # Mediterranean
    collection_boxes["Med1"] = Box(29, 40, 0, 41.5)
    collection_boxes["Med2"] = Box(40, 45, 0, 24)
    collection_boxes["BlckSea1"] = Box(40, 50, 26, 42)  # only for catching the water, not for spreading it
    collection_boxes["CaspSea1"] = Box(35, 50, 46, 55)  # NEW ONE , only for catching
    # Patagonia Atlantic
    collection_boxes["PatAtl1"] = Box(-56.25, -40.0, 290.5, 305)
    # Patagonia Pacific
    collection_boxes["PatPac1"] = Box(-56.25, -36, 282, 290.5)
    collection_boxes["PatPac2"] = Box(-57.5, -56.25, 282, 294.5)
    # New Zealand (South)
    collection_boxes["SNZPac1"] = Box(-47.5, -43.75, 167, 176)
    # New Zealand (North)
    collection_boxes["NNZPac1"] = Box(-43.75, -39, 165, 174.25)
    # Antarctic Ross Sea
    collection_boxes["AARos1"] = Box(-90.0, -68.0, 167.0, 239.0)
    # Antarctic Amundsen Sea
    collection_boxes["AAAmund"] = Box(-90.0, -60.0, 239.0, 297.0)
    # Antarctic Weddell Sea
    collection_boxes["AAWeddell"] = Box(-90.0, -60.0, 297.0, 360.0)
    # Antarctic Riiser-Larson Sea
    collection_boxes["AARiiLar"] = Box(-90.0, -60.0, 0.0, 59)
    # Antarctic Davis Sea
    collection_boxes["AADavis"] = Box(-90.0, -60.0, 59.0, 167.0)

    return collection_boxes


def generate_spreading_regions(cb, um_grid, masked, masked_500m):
    """
    Method to define the collection boxes.
    :param cb: Collection boxes dictionary.
    :param um_grid: Grid of the model.
    :param masked: Land sea mask at the surface.
    :param masked_500m: Land sea mask at 500m.
    :return: List of spreading zones objects.
    """
    
    # Now identify the regions that the water is routed into and spread it over the new larger regions
    us_ecoast = {'name': 'US_East_Coast',
                 'loc': Region([cb["USECoast1"], cb["USECoast2"], cb["USECoast3"], cb["USECoast4"], cb["USECoast5"],
                                cb["USECoast6"], cb["GrLakes1"]], um_grid, masked),
                 'region': Region(
                     [cb["USECoast1"], cb["USECoast2"], cb["USECoast3"], cb["USECoast4"], cb["USECoast4"],
                      cb["USECoast5"]], um_grid,
                     masked_500m)}
    gr_arc = {'name': 'Greenland_Arctic', 'loc': Region([cb["GrArc1"]], um_grid, masked),
              'region': Region([cb["GrArc1"]], um_grid, masked_500m)}
    n_am_arc = {'name': 'N_American_Arctic',
                'loc': Region(
                    [cb["NAMArc1"], cb["NAMArc2"], cb["NAMArc3"], cb["NAMArc4"], cb["NAMArc5"], cb["NWTerr1"],
                     cb["NWTerr2"]], um_grid, masked),
                'region': Region([cb["NAMArc1"], cb["NAMArc2"], cb["NAMArc3"], cb["NAMArc4"]], um_grid, masked_500m)}
    g_o_m = {'name': 'Gulf_of_Mexico', 'loc': Region([cb["GoM1"]], um_grid, masked),
             'region': Region([cb["GoM1"]], um_grid, masked_500m)}
    e_pac = {'name': 'East_Pacific',
             'loc': Region([cb["EPac1"], cb["EPac2"], cb["EPac3"], cb["EPac4"], cb["EPac5"]], um_grid, masked),
             'region': Region([cb["EPac1"], cb["EPac2"], cb["EPac3"], cb["EPac4"], cb["EPac5"]], um_grid,
                              masked_500m)}
    russ_pac = {'name': 'Russia_Pacific', 'loc': Region([cb["RussPac1"]], um_grid, masked),
                'region': Region([cb["RussPac1"]], um_grid, masked_500m)}
    baf_lab = {'name': 'LabradorSea_BaffinBay',
               'loc': Region([cb["BafLab1"], cb["BafLab2"], cb["BafLab3"], cb["HudBay1"], cb["HudBay2"]], um_grid,
                             masked),
               'region': Region([cb["BafLab1"], cb["BafLab2"], cb["BafLab3"]], um_grid, masked_500m)}
    atl_gr = {'name': 'Atlantic_GreenlandIceland', 'loc': Region([cb["AtlGr1"], cb["AtlGr2"]], um_grid, masked),
              'region': Region([cb["AtlGr1"], cb["AtlGr2"]], um_grid, masked_500m)}
    e_gr_ice = {'name': 'EastGreenland_Iceland', 'loc': Region([cb["EGrIce1"], cb["EGrIce2"]], um_grid, masked),
                'region': Region([cb["EGrIce1"], cb["EGrIce2"]], um_grid, masked_500m)}
    e_ice = {'name': 'EastIceland', 'loc': Region([cb["EIceland1"]], um_grid, masked),
             'region': Region([cb["EIceland1"]], um_grid, masked_500m)}
    uk_atl = {'name': 'UK_Atlantic', 'loc': Region([cb["UKAtl1"]], um_grid, masked),
              'region': Region([cb["UKAtl1"]], um_grid, masked_500m)}
    eur_gin = {'name': 'Eurasian_GINSeas', 'loc': Region(
        [cb["EurGIN1"], cb["EurGIN2"], cb["EurGIN3"], cb["EurGIN4"], cb["EurGIN5"], cb["EurGIN6"], cb["Baltic1"],
         cb["Baltic2"]],
        um_grid, masked),
               'region': Region([cb["EurGIN1"], cb["EurGIN2"], cb["EurGIN3"], cb["EurGIN4"], cb["EurGIN5"]], um_grid,
                                masked_500m)}
    s_iceland = {'name': 'South_Iceland', 'loc': Region([cb["SIceland1"]], um_grid, masked),
                 'region': Region([cb["SIceland1"]], um_grid, masked_500m)}
    sib_arc = {'name': 'Siberian_Arctic', 'loc': Region([cb["SibArc1"], cb["SibArc2"]], um_grid, masked),
               'region': Region([cb["SibArc1"]], um_grid, masked_500m)}
    eur_arc = {'name': 'Eurasian_Arctic',
               'loc': Region([cb["EurArc1"], cb["EurArc2"], cb["EurArc3"]], um_grid, masked),
               'region': Region([cb["EurArc1"], cb["EurArc2"]], um_grid, masked_500m)}
    med = {'name': 'Mediterranean',
           'loc': Region([cb["Med1"], cb["Med2"], cb["BlckSea1"], cb["CaspSea1"]], um_grid, masked),
           'region': Region([cb["Med1"], cb["Med2"]], um_grid, masked_500m)}
    pat_atl = {'name': 'Patagonia_Atlantic', 'loc': Region([cb["PatAtl1"]], um_grid, masked),
               'region': Region([cb["PatAtl1"]], um_grid, masked_500m)}
    pat_pac = {'name': 'Patagonia_Pacific', 'loc': Region([cb["PatPac1"], cb["PatPac2"]], um_grid, masked),
               'region': Region([cb["PatPac1"], cb["PatPac2"]], um_grid, masked_500m)}
    nnz_pac = {'name': 'NorthNewZealand_Pacific', 'loc': Region([cb["NNZPac1"]], um_grid, masked),
               'region': Region([cb["NNZPac1"]], um_grid, masked_500m)}
    snz_pac = {'name': 'SouthNewZealand_Pacific', 'loc': Region([cb["SNZPac1"]], um_grid, masked),
               'region': Region([cb["SNZPac1"]], um_grid, masked_500m)}
    aa_ros = {'name': 'Antarctica_RossSea', 'loc': Region([cb["AARos1"]], um_grid, masked),
              'region': Region([cb["AARos1"]], um_grid, masked_500m)}
    aa_amund = {'name': 'Antarctica_AmundsenSea', 'loc': Region([cb["AAAmund"]], um_grid, masked),
                'region': Region([cb["AAAmund"]], um_grid, masked_500m)}
    aa_weddell = {'name': 'Antarctica_WeddellSea', 'loc': Region([cb["AAWeddell"]], um_grid, masked),
                  'region': Region([cb["AAWeddell"]], um_grid, masked_500m)}
    aa_rii_lar = {'name': 'Antarctica_RiiserLarsonSea', 'loc': Region([cb["AARiiLar"]], um_grid, masked),
                  'region': Region([cb["AARiiLar"]], um_grid, masked_500m)}
    aa_davis = {'name': 'Antarctica_DavisSea', 'loc': Region([cb["AADavis"]], um_grid, masked),
                'region': Region([cb["AADavis"]], um_grid, masked_500m)}
    
    return [us_ecoast, gr_arc, n_am_arc, g_o_m, e_pac, russ_pac, baf_lab, atl_gr, e_gr_ice, e_ice, uk_atl, eur_gin,
            s_iceland, eur_arc, sib_arc, med, pat_atl, pat_pac, nnz_pac, snz_pac, aa_ros, aa_amund, aa_weddell,
            aa_rii_lar, aa_davis]


# ----------------------------- #
# ---------- CLASSES ---------- #
# ----------------------------- #


class Box:
    
    def __init__(self, latmin, latmax, lonmin, lonmax):
        self.latmin = latmin
        self.latmax = latmax
        if lonmin < 0:
            lonmin += 360.0
        if lonmax < 0:
            lonmax += 360.0
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.cells_in = None
        self.ocean_in = None
        self.nc = None
        self.no = None
        # self.get_mask(grid,mask)
    
    def get_mask(self, grid, mask):
        """Count ocean grid boxes within the area"""
        # define grid arrays
        lons = grid.lon_center[:]
        lats = grid.lat_center[:]
        ocean_boxes = np.logical_not(mask)
        #
        lats_in = np.logical_and(lats < self.latmax, lats > self.latmin)
        lons_in = np.logical_and(lons < self.lonmax, lons > self.lonmin)
        self.cells_in = np.logical_and(lats_in, lons_in)
        self.ocean_in = np.logical_and(self.cells_in, ocean_boxes)
        self.nc = np.sum(self.cells_in)
        self.no = np.sum(self.ocean_in)
    
    def __repr__(self):
        return str(self.no) + ' ocean cells in the box'

    def cycle_box(self):
        return [[self.lonmin, self.lonmin, self.lonmax, self.lonmax, self.lonmin],
                [self.latmin, self.latmax, self.latmax, self.latmin, self.latmin]]


class Region:
    
    def __init__(self, boxes, grid, mask):
        self.boxes = boxes[:]
        self.grid = grid
        self.grid_mask = np.copy(mask)
        self.mask = None
        self.no = None
        self.totalarea = None
        
        self.get_mask()
        self.calc_area()
    
    def get_mask(self):
        """Count ocean grid boxes within the region"""
        # define grid arrays
        ocean_boxes = np.logical_not(self.grid_mask)
        #
        ocean_in = np.zeros(ocean_boxes.shape)  # start with no box
        for box in self.boxes:
            # add cells from each box
            box.get_mask(self.grid, self.grid_mask)
            ocean_in = np.logical_or(ocean_in, box.ocean_in)
        self.mask = np.copy(ocean_in)
        self.no = np.sum(self.mask)
    
    def calc_area(self):
        """ calculate surface of the region"""
        self.totalarea = np.ma.array(self.grid.area(), mask=np.logical_not(self.mask[:])).sum()
    
    def calc_total_flux(self):
        pass
    
    def __repr__(self):
        return str(self.no) + ' ocean cells in the region'


class Grid:
    
    def __init__(self, lat, lon):
        self.lon_center, self.lat_center = np.meshgrid(lon.center, lat.center)
        self.lon_lower, self.lat_lower = np.meshgrid(lon.lower, lat.lower)
        self.lon_upper, self.lat_upper = np.meshgrid(lon.upper, lat.upper)
    
    def area(self):
        """
        Area of grid cell is
        S(i,j) = R * R *(crad * (lon.upper[i] -  lon.lower[i])) *
                (sin(lat.upper[j]) - sin(lat.lower[j]))
        """
        r = 6371000.0  # radius of Earth (m)
        crad = np.pi / 180.0
        area = r * r * (crad * (self.lon_upper - self.lon_lower)) * \
               (np.sin(crad * self.lat_upper) - np.sin(crad * self.lat_lower))
        area_globe = np.sum(area)
        area_globe_true = 4 * np.pi * r * r
        assert abs(area_globe - area_globe_true) <= area_globe_true * 1e-6
        # print "calculated numerical area is",area_globe,',',100*area_globe/area_globe_true,'% arithmetical value'
        area = np.copy(area)
        return area


class LonAxis:
    """Define longitude axis boundaries
    and deal with wrapping around"""
    
    def __init__(self, lon):
        lon_p = np.roll(lon, -1)  # shifted longitude
        lon_p[-1] += 360
        lon_m = np.roll(lon, 1)
        lon_m[0] -= 360
        lon_lower = lon - (lon - lon_m) / 2.0
        lon_upper = lon + (lon_p - lon) / 2.0
        #
        self.center = lon[:]
        self.lower = lon_lower[:]
        self.upper = lon_upper[:]


class LatAxis:
    """Define latitude axis boundaries
    and overwrite pole boundaries"""
    
    def __init__(self, lat):
        lat_p = np.roll(lat, -1)  # shifted
        lat_m = np.roll(lat, 1)
        lat_lower = lat - (lat - lat_m) / 2.0
        lat_upper = lat + (lat_p - lat) / 2.0
        #
        self.center = lat[:]
        self.lower = lat_lower[:]
        self.upper = lat_upper[:]
        self.lower[0] = -90
        self.upper[-1] = 90


# -------------------------------- #
# ---------- DEPRECATED ---------- #
# -------------------------------- #


def correction_waterfix(correction, wfix, surface_matrix):
    """
    Adding a correction patch (constant value) to the waterfix. DEPRECATED.
    :param correction: Correction patch.
    :param wfix: Waterfix.
    :param surface_matrix: LSM Surface matrix.
    :return:
    """
    d = 1000  # water density
    return np.where(np.isnan(wfix + correction), 0, wfix + correction) / d * surface_matrix
