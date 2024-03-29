import mw_protocol.glac1d_toolbox as tb
import xarray as xr
import numpy as np
import numpy.ma as ma
import os
import datetime


# ------------------------------------ #
# ---------- SAVING METHODS ---------- #
# ------------------------------------ #

def create_output_folder(folder_name):
    """
    Create an output folder at floder_name if it doesn't exist.
    :param folder_name: path of the folder to create.
    :return: None
    """
    dir_name = f"{output_folder}/{folder_name}"
    print("____ Creating directory at ", dir_name)
    try:
        # Create target Directory
        os.mkdir(dir_name)
        print("____ Directory ", dir_name, " created.")
    except FileExistsError:
        print("____ Directory ", dir_name, " already exists.")


def create_dataset(discharge, time, longitude, latitude, title, start_year, end_year, step, mode=None, mode_smooth=None,
                   lsm_name=None, depth=None):
    """
    Create an xarray dataset based on inputs. The waterfix or discharge formats are to be chose with the depth option.
    :param discharge: Meltwater mass flux discharge file [t * lat * lon] numpy array.
    :param time: Time series of the discharge.
    :param longitude: Longitude series of the discharge.
    :param latitude: Latitude series of the discharge.
    :param title: Title to give to the dataset, created with output_names.
    :param start_year: Start year of the dataset in ky.
    :param end_year: End year of the dataset in ky.
    :param step: Time step of the dataset in y.
    :param mode: Routed, Spreaded or patched. Different stages of processing.
    :param mode_smooth: Mode used for routing smoothing algorithm.
    :param lsm_name: Name of the model experiment.
    :param depth: If depth is none, use the discharge format. Else, use the waterfix format with depth value.
    Default is None (discharge format).
    :return: Xarray dataset.
    """
    if depth is None:
        ds = xr.Dataset({'discharge': (('t', 'latitude', 'longitude'), discharge)},
                        coords={'t': time, 'latitude': latitude, 'longitude': longitude})
    else:
        ds = xr.Dataset({'discharge': (('t', 'depth', 'latitude', 'longitude'), discharge)},
                        coords={'t': time, 'depth': depth, 'latitude': latitude, 'longitude': longitude})
    
    ds['discharge'].attrs['units'] = 'kg m-2 s-1'
    ds['discharge'].attrs['longname'] = 'P-E FLUX CORRECTION       KG/M2/S  A'
    
    ds['t'].attrs['long_name'] = 'time'
    ds['t'].attrs['units'] = 'years since 0000-01-01 00:00:00'
    ds['t'].attrs['calendar'] = '360_days'
    
    if depth is not None:  # If it is a waterfix.
        ds['depth'].attrs['long_name'] = 'depth'
    
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
    
    ds.attrs['title'] = title
    if depth is None:  # If it is not a waterfix.
        ds.attrs['start_year'] = start_year
        ds.attrs['end_year'] = end_year
        ds.attrs['step'] = step
        if mode is not None:
            ds.attrs['mode'] = mode
        if mode_smooth is not None:
            ds.attrs['mode_smooth'] = mode_smooth
        if lsm_name is not None:
            ds.attrs['lsm'] = lsm_name
    ds.attrs['history'] = f"Created {datetime.datetime.now()} by Yvan Romé"
    
    return ds


def output_names(start_year, end_year, step, mode, mode_smooth, lsm_name, file_name='discharge'):
    """
    Compute the output names from the options.
    :param start_year: Start year of the dataset in ky.
    :param end_year: End year of the dataset in ky.
    :param step: Time step of the dataset in y.
    :param mode: Routed, Spreaded or patched. Different stages of processing.
    :param mode_smooth: Mode used for routing smoothing algorithm.
    :param lsm_name: Name of the model experiment.
    :param file_name: Discharge or waterfix file.
    :return:
    """
    if file_name == 'wfix':
        file_description = 'Waterfix'
    else:
        file_description = 'Meltwater discharge'
    
    file_path = f"{lsm_name}.{file_name}.glac_mw.nc"
    
    if mode == "routed":
        folder_path = f"glac.{start_year}_{end_year}_{step}.{mode_smooth}"
        title = f"{file_description} for transient GLAC1D last delgaciation HadCM3 simulations " \
                f"- {lsm_name} land sea mask - {start_year}kya to {end_year}kya with {step}yrs time step " \
                f"- {mode_smooth} mode processing - spreading applied but no patch correction."
    elif mode == "spreaded":
        folder_path = f"glac.{start_year}_{end_year}_{step}.{mode_smooth}_s"
        title = f"{file_description} for transient GLAC1D last delgaciation HadCM3 simulations " \
                f"- {lsm_name} land sea mask - {start_year}kya to {end_year}kya with {step}yrs time step " \
                f"- {mode_smooth} mode processing - spreading applied but no patch correction."
    elif mode == "patched":
        folder_path = f"glac.{start_year}_{end_year}_{step}.{mode_smooth}_sp"
        title = f"{file_description} for transient GLAC1D last delgaciation HadCM3 simulations " \
                f"- {lsm_name} land sea mask - {start_year}kya to {end_year}kya with {step}yrs time step " \
                f"- {mode_smooth} mode processing - spreading and patch correction applied."
    else:
        print("The mode wasn't recognized.")
        raise ValueError("Invalid mode.")
    
    return folder_path, file_path, title


def create_corrected_waterfix(path_ref, experiment_name, lsm_name, ds_wfix, start_year, end_year, sav_path=None):
    """
    Create corrected waterfix
    :param path_ref: Path of the experiment to extract the patch from.
    :param experiment_name: Name of the experiment
    :param lsm_name: Name of the land sea mask used in the experiment/waterfix
    :param ds_wfix: Waterfix of the reference experiment.
    :param start_year: Start year of the dataset in ky.
    :param end_year: End year of the dataset in ky.
    :param sav_path: Path of output file.
    :return:
    """
    ds_new = ds_wfix.copy()
    ds_new.field672.values[0, 0, :, :-2] += calculate_patch(f"{path_ref}/{experiment_name}", ds_wfix, start_year,
                                                            end_year)
    ds_new.attrs['title'] = f"Waterfix for HadCM3 simulations [GLAC-1D] - " \
                            f"Corrected using {experiment_name} salinity drift between years {start_year} " \
                            f"and {end_year} - Land sea mask : {lsm_name}"
    ds_new.attrs['history'] = f"Created on {datetime.datetime.now()} by Yvan Romé"
    
    if sav_path is not None:
        ds_new.to_netcdf(sav_path)
    
    return ds_new


def create_updated_waterfix(discharge, index, year, lsm_name, ds_wfix, sav_path=None):
    """
    Create corrected waterfix
    :param path_ref: Path of the experiment to extract the patch from.
    :param experiment_name: Name of the experiment
    :param lsm_name: Name of the land sea mask used in the experiment/waterfix
    :param ds_wfix: Waterfix of the reference experiment.
    :param start_year: Start year of the dataset in ky.
    :param end_year: End year of the dataset in ky.
    :param sav_path: Path of output file.
    :return:
    """
    
    if os.path.isfile(sav_path):
        os.remove(sav_path)
        
    ds_new = ds_wfix.copy()
    ds_new.field672.values[0, 0, :, :-2] += discharge
    
    ds_new.attrs['title'] = f"Waterfix for HadCM3 simulations [GLAC-1D] - " \
                            f"Created using value i={index} of GLAC-1D discharge corresponding to year {year} " \
                            f"- Land sea mask : {lsm_name}"
    ds_new.attrs['history'] = f"Created on {datetime.datetime.now()} by Yvan Romé"
    
    if sav_path is not None:
        ds_new.to_netcdf(sav_path)
    
    return ds_new


def discharge2input(discharge, ds_lsm, ds_wfix, experiment_name, lsm_name, mode, start_year, end_year, step,
                    sav_name, sav_folder, mode_smooth="diff", time_shift=0):
    """
    Fast function to turn a discharge array into a file in the waterfix format.

    Parameters
    ----------
    ds_wfix: xarray.Dataset
        Waferfix dataset (for coordinates)
    start_year: int
        First year of the dataset in ky. (Indicate manually negative values)
    end_year: int
        Laste year of the dataset in ky. (Indicate manually negative values)
    step: int
        Time step in y
    mode: str
        Routed, Spreaded or Patched.
    lsm_name: str
        Name of the lsm experiment
    sav_folder: str
        Path to the save folder
    sav_name: str
        Name of the save file
    ds_lsm: xarray.Dataset
        Xarray dataset with the lsm values
    experiment_name: str
        Name of the HadCM3 experiment for which this waterfix was initially created.
    discharge: np.ndarray
        numpy array discharge values
    mode_smooth: str, optional
        Name of the smoothing algorithm. ex: diff
    time_shift: int
        In case the dates have to be shifted from their initial values
    """
    
    longitude, latitude = ds_wfix.longitude.values[:-2], ds_wfix.latitude.values
    time = np.arange(start_year * 1000, end_year * 1000 + step, step) + time_shift
    lsm = ds_lsm.lsm.values

    # m3/s to kg/m2/s
    processed_discharge = m3s_to_kgm2s(discharge, longitude, latitude)

    title = f"Waterfix initially created for {experiment_name} HadCM3 experiment." \
            f"- {lsm_name} land sea mask - {start_year}ky to {end_year}ky with {step}yrs time step " \
            f"- {mode_smooth} processing - {mode} algorithm applied."
    
    ds = create_dataset(masking_method(processed_discharge, lsm), time, longitude, latitude, title,
                        start_year, end_year, step, mode, mode_smooth, lsm_name)
    
    sav_path = f"{sav_folder}/{sav_name}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)
    
    return ds


def process_discharge_time(ds_ref, ds_lsm, experiment_name, sav_name, sav_folder,
                           new_start_year=None, new_end_year=None, new_step=None, time_shift=0):
    longitude, latitude = ds_ref.longitude.values, ds_ref.latitude.values
    lsm_name, mode, mode_smooth = ds_ref.lsm, ds_ref.mode, ds_ref.mode_smooth
    step_ref, start_ref, end_ref = ds_ref.step, ds_ref.start_year, ds_ref.end_year
    lsm = ds_lsm.lsm.values
    
    if new_step is not None and new_start_year is not None and new_end_year is not None:
        raise AttributeError("Impossible to correct the time step and the start/end years at the same time."
                             "Please do it separatly.")
    elif new_step is not None:
        processed_discharge, processed_time = process_step(ds_ref, new_step)
        start_year, end_year = start_ref, end_ref
        step = new_step
    elif new_start_year is not None and new_end_year is not None:
        processed_discharge, processed_time = process_time(ds_ref, new_start_year, new_end_year, time_shift=time_shift)
        start_year = new_start_year if new_start_year is not None else start_ref
        end_year = new_end_year if new_end_year is not None else end_ref
        step = step_ref
    else:
        raise AttributeError("Mode not recognized.")
    
    title = f"Waterfix initially created for {experiment_name} HadCM3 experiment." \
            f"- {lsm_name} land sea mask - {start_year}ky to {end_year}ky with {step}yrs time step " \
            f"- {mode_smooth} processing - {mode} algorithm applied."
    
    ds = create_dataset(masking_method(processed_discharge, lsm), processed_time, longitude, latitude, title,
                        start_year, end_year, step, mode, mode_smooth, lsm_name)
    
    sav_path = f"{sav_folder}/{sav_name}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)
    
    return ds


def create_input(ds_discharge, ds_lsm, experiment_name, lsm_name, mode, start_year, end_year, step,
                 sav_name, sav_folder, mode_smooth="diff"):
    """
    Fast function to turn a discharge file into a a file to be converted in the right format.
    
    Parameters
    ----------
    start_year: int
        First year of the dataset in ky. (Indicate manually negative values)
    end_year: int
        Laste year of the dataset in ky. (Indicate manually negative values)
    step: int
        Time step in y
    mode: str
        Routed, Spreaded or Patched.
    lsm_name: str
        Name of the lsm experiment
    sav_folder: str
        Path to the save folder
    sav_name: str
        Name of the save file
    ds_lsm: xarray.Dataset
        Xarray dataset with the lsm values
    experiment_name: str
        Name of the HadCM3 experiment for which this waterfix was initially created.
    ds_discharge: xarray.Dataset
        Xarray dataset with the discharge values
    mode_smooth: str, optional
        Name of the smoothing algorithm. ex: diff
    """
    
    longitude, latitude = ds_discharge.longitude.values, ds_discharge.latitude.values
    time = np.arange(start_year * 1000, end_year * 1000 + step, step)
    lsm = ds_lsm.lsm.values
    
    title = f"Waterfix initially created for {experiment_name} HadCM3 experiment." \
            f"- {lsm_name} land sea mask - {start_year}kya to {end_year}kya with {step}yrs time step " \
            f"- {mode_smooth} processing - {mode} algorithm applied."
    
    ds = create_dataset(masking_method(ds_discharge.discharge.values, lsm), time, longitude, latitude, title,
                        start_year, end_year, step, mode, mode_smooth, lsm_name)
    
    sav_path = f"{sav_folder}/{sav_name}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)
    
    return ds


# ---------------------------------------- #
# ---------- CONVERSION METHODS ---------- #
# ---------------------------------------- #

def masking_method(discharge, lsm):
    """
    Mask a discharge array using a land sea mask.
    :param discharge: [t*lat*lon] numpy array.
    :param lsm: [lat*lon] numpy array.
    :return: Masked discharge array [t*lat*lon]
    """
    lsm_3d = np.resize(lsm, discharge.shape)
    return ma.array(discharge, mask=lsm_3d)


def m3s_to_kgm2s(discharge, lon, lat):
    """
    Convert a discharge volume flux to a discharge surface mass flux.
    :param discharge: [t*lat*lon] numpy array.
    :param lon: Longitude series of discharge.
    :param lat: Latitude series of discharge.
    :return: Discharge surface mass flux [t*lat*lon] numpy array.
    """
    d = 1000  # water density
    return np.divide(discharge * d, tb.surface_matrix(lon, lat))


def kgm2s_to_m3s(discharge, lon, lat):
    """
    Convert a discharge surface mass flux a discharge volume flux.
    :param discharge: [t*lat*lon] numpy array.
    :param lon: Longitude series of discharge.
    :param lat: Latitude series of discharge.
    :return: Discharge volume flux [t*lat*lon] numpy array.
    """
    
    d = 1000  # water density
    return np.multiply(discharge / d, tb.surface_matrix(lon, lat))


# ---------------------------------------- #
# ---------- PROCESSING METHODS ---------- #
# ---------------------------------------- #

def process_time(ds_ref, start, end, discharge_in=None, time_shift=0):
    """
    Process time of a reference xarray dataset to fit new start and end years.
    If a discharge file is specified, process the discharge file instead.
    :param ds_ref: Reference xarray dataset to be processed.
    :param start: new start year in ky.
    :param end: new end year in ky.
    :param discharge_in: Optional discharge file to be processed [t*lat*lon] numpy array.
    :return: processed discharge [t*lat*lon] numpy array and new corresponding time series.
    """
    discharge_ref, t_ref = ds_ref.discharge.values, ds_ref.t.values
    start_ref, end_ref, step_ref = ds_ref.start_year, ds_ref.end_year, ds_ref.step
    
    discharge = discharge_in if discharge_in is not None else discharge_ref
    
    if len(discharge) == 1:
        print("____ Impossible to process time for a single step discharge array.")
        return discharge, t_ref
    
    start_k, end_k = start * 1000 + time_shift, end * 1000 + time_shift
    processed_time = np.arange(start_k, end_k + 100, step_ref)
    n_t, n_lat, n_lon = discharge_ref.shape
    discharge_processed = np.zeros((len(processed_time), n_lat, n_lon))
    
    if (start >= start_ref) and (end <= end_ref):
        id_start = np.where(t_ref == start_k)[0][0]
        id_end = np.where(t_ref == end_k)[0][0]
        
        discharge_processed[:] = discharge_ref[id_start:id_end + 1]
    
    elif (start < -26) and (end <= 0):
        id_26 = np.where(processed_time == -26000)[0][0]
        id_end = np.where(t_ref == end_k)[0][0]
        
        discharge_processed[:id_26] = discharge_ref[0]
        discharge_processed[id_26:] = discharge_ref[:id_end]
    
    elif (start >= -26) and (end > 0):
        id_start = np.where(t_ref == start_k)[0][0]
        id_0 = np.where(processed_time == 0)[0][0]
        
        discharge_processed[:id_0] = discharge_ref[id_start:]
        discharge_processed[id_0:] = discharge_ref[-1]
    
    elif (start < -26) and (end > 0):
        id_26 = np.where(processed_time == -26000)[0][0]
        id_0 = np.where(processed_time == 0)[0][0]
        
        discharge_processed[:id_26] = discharge_ref[0]
        discharge_processed[id_26: id_0 + 1] = discharge_ref[:]
        discharge_processed[id_0:] = discharge_ref[-1]
    
    else:
        raise ValueError("!!! Start or end paramters incorect")
    
    return discharge_processed, processed_time


def process_step(ds_ref, new_step):
    """
    Process time step of a reference xarray dataset to fit a new time step.
    :param ds_ref: Reference xarray dataset to be processed.
    :param new_step: new time step in y.
    :return: processed discharge [t*lat*lon] numpy array and new corresponding time seriess.
    """
    discharge_ref, t_ref = ds_ref.discharge.values, ds_ref.t.values
    start_ref, end_ref, step_ref = ds_ref.start_year * 1000, ds_ref.end_year * 1000, ds_ref.step
    n_t, n_lat, n_lon = discharge_ref.shape
    
    if new_step < step_ref:
        raise ValueError("The new step is smaller than the old one.")
    elif new_step // step_ref != new_step / step_ref:
        raise ValueError("The new step should be a multiple of the old one.")
    else:
        inc = new_step // step_ref
    processed_time = np.arange(start_ref, end_ref + new_step, new_step)
    discharge_processed = np.zeros((len(processed_time), n_lat, n_lon))
    
    discharge_processed[::] = discharge_ref[::inc]
    
    return discharge_processed, processed_time


# -------------------------------------------- #
# ---------- PATCH/WATERFIX METHODS ---------- #
# -------------------------------------------- #


def calculate_patch(path, ds_wfix, start_date, end_date):
    """
    Calculate a drift correction patch based on the output of a reference experiment.
    :param path: Path of the experiment to extract the patch from. Format: path/exp
    :param ds_wfix: Waterfix of the reference experiment.
    :param start_date: Start year of the dataset in ky.
    :param end_date: End year of the dataset in ky.
    :return:
    """
    print(f"____ Computation of the drift patch")
    wfix = ds_wfix.field672.isel(depth=0).isel(t=0).values[:, :-2]
    srf_sal_flux = np.zeros(wfix.shape)
    
    for year in np.arange(start_date, end_date, 1):
        ds = xr.open_dataset(f'{path}o#pg00000{year}c1+.nc')
        srf_sal_flux += ds.srfSalFlux_ym_uo_1.isel(t=0).isel(unspecified=0).values - wfix
    
    return srf_sal_flux / (end_date - start_date)


def discharge_to_waterfix(discharge, longitude):
    """
    Add two points of longitude and a coordinate of depth to a discharge array to fit the waterfix format.
    :param discharge: [t*lat*lon] numpy array.
    :param longitude: Corresponding longitude series.
    :return: [t*depth*lat*lon+2] numpy array.
    """
    n_t, n_lat, n_lon = discharge.shape
    
    processed_longitude = longitude
    
    processed_discharge = np.zeros((n_t, 1, n_lat, n_lon + 2))
    processed_discharge[:, 0, :, 0:n_lon] = discharge
    processed_discharge[:, 0, :, n_lon:] = discharge[:, :, 0:2]
    
    return processed_discharge, processed_longitude


# --------------------------------- #
# ---------- OLD METHODS ---------- #
# --------------------------------- #

# OUTDATED!!!

# Change for a new user
output_folder = "/nfs/annie/eeymr/work/outputs/proj_glac_mw"


def saving(discharge, ds_lsm, lsm_name, mode, start_year=-26, end_year=0, step=100, mode_smooth="diff"):
    """
    Save a discharge file in the output_folder. Create the directory, convert the discharge and create an xarray
    dataset based on the options. This method is to be used in first instance, but the correcting method yield the
    same result bby cropping an existing dataset.
    :param discharge: Meltwater mass surface flux discharge file [t * lat * lon] numpy array (kg/m2/s)
    :param ds_lsm: Xarray dataset of the corresponding lsm (for masking operation)
    :param lsm_name: Name of the base experiment
    :param mode: Routed, Spreaded or patched. Different stages of processing.
    :param start_year: Start year of the dataset in ky. Default is -26ky.
    :param end_year: End year of the dataset in ky. Default is 0ky.
    :param step: Time step of the dataset in y. Default is 100y.
    :param mode_smooth: Mode used for routing smoothing algorithm.
    :return: None
      """
    
    print("__ Saving algorithm")
    
    lsm, longitude, latitude = ds_lsm.lsm.values, ds_lsm.longitude.values, ds_lsm.latitude.values
    
    folder_path, file_path, title = output_names(start_year, end_year, step, mode, mode_smooth, lsm_name)
    
    create_output_folder(folder_path)
    
    # m3/s to kg/m2/s
    processed_discharge = m3s_to_kgm2s(discharge, longitude, latitude)
    
    # masked array
    masked_discharge = masking_method(processed_discharge, lsm)
    
    # time
    time = np.arange(start_year * 1000, end_year * 1000 + step, step)
    
    ds = create_dataset(masked_discharge, time, longitude, latitude, title, start_year, end_year, step,
                        mode, mode_smooth, lsm_name)
    
    sav_path = f"{output_folder}/{folder_path}/{file_path}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)


def correcting(ds_ref, new_start_year=None, new_end_year=None, new_step=None, time_shift=0):
    """
    Save a discharge file in the output_folder from an existing dataset.
    Create the directory, extract and convert the discharge and create an xarray dataset based on the reference.
    :param ds_ref: Xarray dataset created by a glac_mw method to be processed.
    :param new_start_year: New start year of the dataset. If the start year is smaller than the old one, the first value
     of the dataset will be copied on the new years. new_start_year and new_end_year have to be filled at the same time.
    :param new_end_year: New end year of the dataset. If the end year is bigger than the old one, the last value
     of the dataset will be copied on the new years. new_start_year and new_end_year have to be filled at the same time.
    :param new_step: New time step of the dataset. Has to be a multpile of the old one.
    :return: None
    """
    print("__ Correction algorithm")
    
    longitude, latitude = ds_ref.longitude.values, ds_ref.latitude.values
    lsm_name, mode, mode_smooth = ds_ref.lsm, ds_ref.mode, ds_ref.mode_smooth
    step_ref, start_ref, end_ref = ds_ref.step, ds_ref.start_year, ds_ref.end_year
    
    if new_step is not None and new_start_year is not None and new_end_year is not None:
        raise AttributeError("Impossible to correct the time step and the start/end years at the same time."
                             "Please do it separatly.")
    elif new_step is not None:
        processed_mw, processed_time = process_step(ds_ref, new_step)
        new_start_year, new_end_year = start_ref, end_ref
    elif new_start_year is not None and new_end_year is not None:
        processed_mw, processed_time = process_time(ds_ref, new_start_year, new_end_year, time_shift)
        new_step = step_ref
    else:
        raise AttributeError("Mode not recognized.")
    
    folder_path, file_path, title = output_names(new_start_year, new_end_year, new_step, mode,
                                                 mode_smooth, lsm_name)
    
    create_output_folder(folder_path)
    
    ds = create_dataset(processed_mw, processed_time, longitude, latitude, title, new_start_year, new_end_year,
                        new_step, mode, mode_smooth, lsm_name)
    
    sav_path = f"{output_folder}/{folder_path}/{file_path}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)


def to_waterfix(ds_ref, ds_wfix):
    """
    Convert an existing dataset created from a glac_mw method to a waterfix format.
    :param ds_ref: Existing discharge dataset.
    :param ds_wfix: Model waterfix xarray dataset.
    :return: None
    """
    print("__ To waterfix algorithm")
    longitude, latitude, depth = ds_wfix.longitude.values, ds_wfix.latitude.values, ds_wfix.depth.values
    
    discharge, time = ds_ref.discharge.values, ds_ref.t.values
    start_year, end_year, step, mode, mode_smooth, lsm_name = \
        ds_ref.start_year, ds_ref.end_year, ds_ref.step, ds_ref.mode, ds_ref.mode_smooth, ds_ref.lsm
    
    folder_path, file_path, title = output_names(start_year, end_year, step, mode, mode_smooth, lsm_name,
                                                 file_name='wfix')
    
    processed_discharge, processed_longitude = discharge_to_waterfix(discharge, longitude)
    
    ds = create_dataset(processed_discharge, time, processed_longitude, latitude, title, start_year, end_year, step,
                        mode, mode_smooth, lsm_name, depth=depth)
    
    sav_path = f"{output_folder}/{folder_path}/{file_path}"
    print(f"__ Saving at: {sav_path}")
    ds.to_netcdf(sav_path)
