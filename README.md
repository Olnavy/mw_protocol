# Meltwater Discharge Protocol

Algorithm to create meltwater discharge fields dervied from an ice sheet reconstruction. Can create point-wise or spread discharge. Tested on HadcM3/Famous and Nemo ocean grids. 

Adapted from the algorithm descibed in [Romé et al. 2022](https://doi.org/10.1029/2022PA004451).

To create the discharge files, create a copy of the repository and modify create\_point\_discharge.py or create\_spread\_discharge.py with your desired inputs and options, and run them. You will need to have the right dependencies to do this (see bellow). You will also have to modify the collection boxes/spreading regions in the inputs to suit your model and land-sea-mask. You can use the visual\_tools.ipynb to do so. If you want to adapt or debug the algorithm, you can find a detail of each step in the algorithm/ipynb notebook.

Python dependecies:
 - Jupyter
 - Xarray
 - Numpy
 - XESMF
 - Matplotlib
 - Cartopy
 - Shapely

Please also make sure that the mw\_protocol folder is added to your python PATH.

