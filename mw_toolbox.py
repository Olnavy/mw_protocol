import numpy as np

earth_radius = 6371000  # Earth's radius in meters

def calculate_cell_area(lat1, lat2, lon1, lon2, radius=earth_radius):
    """
    Calculate the area of a single grid cell on a sphere.
    
    :param lat1: Latitude of the first boundary (in degrees)
    :param lat2: Latitude of the second boundary (in degrees)
    :param lon1: Longitude of the first boundary (in degrees)
    :param lon2: Longitude of the second boundary (in degrees)
    :param radius: Radius of the sphere (default is Earth's radius in meters)
    :return: Area of the grid cell (in square meters)
    """
    # Convert latitudes to radians
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)

    return 2 * np.pi * radius ** 2 * np.abs(np.sin(lat2_rad) - np.sin(lat1_rad)) * np.abs(lon2 - lon1) / (2 * np.pi)


def calculate_surface_matrix(longitudes:np.array, latitudes:np.array):
    """
    Create a surface matrix for a grid defined by latitudes and longitudes.
    
    :param latitudes: 1D array of latitude values (in degrees)
    :param longitudes: 1D array of longitude values (in degrees)
    :return: 2D array of grid cell areas (in square meters)
    """
    print("____ Calculating surface matrix")
    
    # Calculate latitude and longitude step sizes
    lat_step = np.diff(latitudes)
    lon_step = np.diff(longitudes)
    
    # Handle non-regular grids by using the midpoint for each cell
    lat_step = np.append(lat_step, lat_step[-1])  # Extend last step for edge case
    lon_step = np.append(lon_step, lon_step[-1])  # Extend last step for edge case
    
    # Initialize a 2D array for cell areas
    surface_matrix = np.zeros((len(latitudes), len(longitudes)))
    
    # Calculate the area for each grid cell
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            # Define the latitude and longitude boundaries for the cell
            if i < len(latitudes) - 1:
                lat1, lat2 = lat, lat + lat_step[i]
            else:
                lat1, lat2 = lat - lat_step[i - 1], lat
            
            if j < len(longitudes) - 1:
                lon1, lon2 = lon, lon + lon_step[j]
            else:
                lon1, lon2 = lon - lon_step[j - 1], lon
            
            # Calculate the area of the cell
            surface_matrix[i, j] = calculate_cell_area(lat1, lat2, lon1, lon2)
    
    return surface_matrix