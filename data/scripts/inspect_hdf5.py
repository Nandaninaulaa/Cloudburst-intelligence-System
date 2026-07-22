import h5py
import numpy as np

file_path = r"C:\Users\bohra\Downloads\GPM_3IMERGHH_07-20251111_162614\3B-HHR.MS.MRG.3IMERG.20210101-S000000-E002959.0000.V07B.HDF5"  # change this

lat_min, lat_max = 27.5, 32.5
lon_min, lon_max = 76.5, 82.5

with h5py.File(file_path, 'r') as f:
    grid = f['Grid']
    lat = grid['lat'][:]
    lon = grid['lon'][:]
    
    # Find points within or near the Uttarakhand box
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    
    print(f"Total lat points in range: {lat_mask.sum()}")
    print(f"Total lon points in range: {lon_mask.sum()}")
    
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        print(" No coordinate grid cells inside bounding box — maybe dataset resolution is coarse.")
        print("Closest latitudes to Uttarakhand:")
        print(lat[np.abs(lat - 30).argsort()[:10]])
        print("Closest longitudes to Uttarakhand:")
        print(lon[np.abs(lon - 79).argsort()[:10]])
