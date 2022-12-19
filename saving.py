import mw_protocol.toolbox as tb
import xarray as xr
import numpy as np
import numpy.ma as ma
import os
import datetime

# ---------------------------------- #
# ---------- MAIN METHODS ---------- #
# ---------------------------------- #

def discharge_to_ancil(ds_input, ds_lsm):
    """
    Convert to kg/m2/s, add surface depth and mask lsm.
    """

    print("__ Converting discharge to ancil")

    ds_output = ds_input.copy()
    ds_output = mask_lsm(add_depth(m3s_to_kgm2s(ds_output)), ds_lsm)

    ds_output.attrs['title'] = "Spread meltwater + waterfix mask for transient last delgaciation HadCM3 project"
    ds_output.attrs['history'] = "Created on " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " by Yvan Malo Romé"

    return ds_output


def ancil_to_discharge(ds_input):
    """
    Convert to m3/s and remove surface depth.
    """

    print("__ Converting ancil to discharge")

    ds_output = ds_input.copy()
    ds_output = unmask_lsm(remove_depth(kgm2s_to_m3s(ds_output)))

    return ds_output


def discharge_to_waterfix(ds_input):
    """
    Convert to kg/m2/s, loop the longitudes, add surface depth and rename discharge to field672.
    """

    print("__ Converting discharge to waterfix")

    ds_output = ds_input.copy()
    ds_output = add_depth(loop_longitude(m3s_to_kgm2s(ds_output))).rename({'discharge':'field672'})

    return ds_output

def waterfix_to_discharge(ds_input):
    """
    Convert to m3/s, unloop the longitudes, remove surface depth and rename field672 to discharge.
    """

    print("__ Converting waterfix to discharge")

    ds_output = ds_input.copy()
    ds_output = remove_depth(unloop_longitude(kgm2s_to_m3s(ds_output.rename({'field672':'discharge'}))))

    return ds_output


# ---------------------------------------- #
# ---------- PROCESSING METHODS ---------- #
# ---------------------------------------- #

def loop_longitude(ds_input):
    """
    Copy the first two longitude slices to loop the longitude field.
    """

    print("____ looping longitude")
    new_longitude = ds_input.longitude[:2]+360
    return xr.concat([ds_input, ds_input.isel(longitude=slice(0,2)).assign_coords({"longitude":new_longitude})], dim="longitude")


def unloop_longitude(ds_input):
    """
    Remove the last two longitude slices.
    """

    print("____ unlooping longitude")
    return ds_input.isel(longitude=slice(None,-2))


def m3s_to_kgm2s(ds_input):
    """
    Convert a discharge dataset from m3/s to kg/m2/s.
    """

    print("____ Converting m3/s to kg/m2/s")
    d = 1000  # water density
    ds_output = ds_input.copy().update({"discharge": np.divide(ds_input.discharge * d, tb.surface_matrix(ds_input.longitude.values, ds_input.latitude.values))})
    ds_output['discharge'].attrs['units'] = "kg/m2/s"
    return  ds_output


def kgm2s_to_m3s(ds_input):
    """
    Convert a discharge dataset from kg/m2/s to m3/s.
    """
    
    print("____ Converting kg/m2/s to m3/s")
    d = 1000  # water density
    ds_output = ds_input.copy().update({"discharge": np.multiply(ds_input.discharge / d, tb.surface_matrix(ds_input.longitude.values, ds_input.latitude.values))})
    ds_output['discharge'].attrs['units'] = "m3/s"
    return  ds_output


def crop_years(ds_input, new_start_year=None, new_end_year=None):
    """
    Change the start years and/or end years of a discharge dataset.
    """

    print("____ Cropping years")

    start_ref, end_ref = ds_input.start_year, ds_input.end_year
    start = new_start_year if new_start_year else start_ref
    end = new_end_year if new_end_year else end_ref

    assert start>=start_ref and start<=end_ref, "____ The new start year has to be between the old start and end years"
    assert end<=end_ref and end>=start_ref, "_____ The new start year has to be between the old start and end years"
    assert len(ds_input.t) != 1, "____ Impossible to process time for a single step discharge array."

    ds_output = ds_input.copy().sel(t=slice(new_start_year,new_end_year))
    ds_output.attrs['start_year'] = np.min(ds_output.t.values)
    ds_output.attrs['end_year'] = np.max(ds_output.t.values)

    return ds_output


def multiply_steps(ds_input, multiplier):
    """
    Multiply the time steps of a discharge dataset.
    """

    print("____ Multiplying time steps")

    assert (isinstance(multiplier, int) or multiplier is None)

    ds_output = ds_input.copy().sel(t=slice(None,None, multiplier))
    ds_output.attrs['step'] = ds_input.step*multiplier if multiplier is not None else ds_input.step

    return ds_output


def add_depth(ds_input, depth_value=5.0, depth_name='depth'):
    """
    Convert a discharge dataset from kg/m2/s to m3/s.
    """

    print("____ Adding depth coordinate")

    ds_output = ds_input.expand_dims(depth_name, axis=1).assign_coords({depth_name:[depth_value]})
    ds_output[depth_name].attrs['units'] = 'm'
    ds_output[depth_name].attrs['positive'] = 'down'

    return ds_output


def remove_depth(ds_input, depth_name='depth'):
    """
    Convert a discharge dataset from kg/m2/s to m3/s.
    """

    print("____ Removing depth coordinate")

    if depth_name in ds_input.coords.keys():
        return ds_input.isel({depth_name:0}).drop(depth_name)
    elif 'depth' in ds_input.coords.keys():
        return ds_input.isel({'depth':0}).drop('depth')
    elif 'unspecified' in ds_input.coords.keys():
        return ds_input.isel({'unspecified':0}).drop('unspecified')
    else:
        print("__ No depth coordinate was found")
        return ds_input


def mask_lsm(ds_input, ds_lsm):

    print("____ Masking land-sea mask")
    lsm_3d = np.resize(np.where(ds_lsm.lsm==1, np.nan, 1), ds_input.discharge.shape)
    ds_output = ds_input.copy().update({"discharge": ds_input.discharge*lsm_3d})

    if 'name' in ds_lsm.attrs:
        ds_output.attrs['lsm'] = ds_lsm.name
    else:
        print("_____ ** Please add a name to the land-sea mask dataset unsing saving.add_lsm_name")
        ds_output.attrs['lsm'] = "Unknown"

    return ds_output
    

def unmask_lsm(ds_input):

    print("____ Unmasking land-sea mask")


    ds_output = ds_input.copy()
    ds_output.discharge.values = np.where(np.isnan(ds_output.discharge), 0, ds_output.discharge)
    ds_output.attrs['lsm'] = None

    return ds_output


def add_lsm_name(ds_lsm, lsm_name):
    ds_output = ds_lsm.copy()
    ds_output.attrs['name'] = lsm_name

    return ds_output


# ------------------------------------ #
# ---------- SAVING METHODS ---------- #
# ------------------------------------ #

def create_output_directory(dir_name):
    """
    Create an output folder if it doesn't already exist.
    :param folder_name: path of the new folder.
    :return: None
    """
    print("____ Attempting to create a directory at ", dir_name)
    try:
        # Create target Directory
        os.mkdir(dir_name)
        print("____ Directory ", dir_name, " created.")
    except FileExistsError:
        print("____ Directory ", dir_name, " already exists.")


def create_omask(dataset):
    """
    Create an ocean mask from an ocean um da file.
    """
    
    def get_depth(array, depth):
        """
        Derive a depthdepth array from a depthlevel array
        """
        out_array = array.copy()
        out_array.values= np.full(out_array.shape, np.nan)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i,j] >= 0:
                    out_array[i,j] = depth[array[i,j]-1]
        return out_array
    
    # Calculated from the shallowest ocean slice, 1 is land, 0 is ocean.
    lsm = xr.where(dataset.isel(depth=0).isnull(), 1, 0).rename('lsm').drop('depth')
    # Calculated by summing the non empty cells.
    depthlevel = xr.where(dataset.isnull(), np.nan, 1).sum('depth', skipna=True).astype(int)
    depthlevel = xr.where(depthlevel>0, depthlevel, np.nan)
    # Calculated from the dethlevel
    depthdepth = get_depth(depthlevel.astype(int), dataset.depth)
    
    ds = lsm.rename('lsm').to_dataset()
    ds['depthlevel'] = depthlevel
    ds['depthdepth'] = depthdepth
    ds.attrs['title'] = f"Produced using mw_protocol by Olnavy from {dataset.name} file"
    
    return ds

# -------------------------------------------- #
# ---------- PATCH/WATERFIX METHODS ---------- #
# -------------------------------------------- #

# DEPRECATED!!!

# def calculate_patch(path, ds_wfix, start_date, end_date):
#     """
#     Calculate a drift correction patch based on the output of a reference experiment.
#     :param path: Path of the experiment to extract the patch from. Format: path/exp
#     :param ds_wfix: Waterfix of the reference experiment.
#     :param start_date: Start year of the dataset in ky.
#     :param end_date: End year of the dataset in ky.
#     :return:
#     """
#     print(f"____ Computation of the drift patch")
#     wfix = ds_wfix.field672.isel(depth=0).isel(t=0).values[:, :-2]
#     srf_sal_flux = np.zeros(wfix.shape)
    
#     for year in np.arange(start_date, end_date, 1):
#         ds = xr.open_dataset(f'{path}o#pg00000{year}c1+.nc')
#         srf_sal_flux += ds.srfSalFlux_ym_uo_1.isel(t=0).isel(unspecified=0).values - wfix
    
#     return srf_sal_flux / (end_date - start_date)
