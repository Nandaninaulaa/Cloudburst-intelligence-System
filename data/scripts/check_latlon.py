import h5py
import numpy as np

file_path = r"c:\Users\bohra\Downloads\GPM_3IMERGHH_07-20251111_162614\3B-HHR.MS.MRG.3IMERG.20210101-S000000-E002959.0000.V07B.HDF5"  # change to any one file path

with h5py.File(file_path, 'r') as f:
    grid = f['Grid']
    lat = grid['lat'][:]
    lon = grid['lon'][:]

    print("Latitude range:", np.min(lat), "to", np.max(lat))
    print("Longitude range:", np.min(lon), "to", np.max(lon))
