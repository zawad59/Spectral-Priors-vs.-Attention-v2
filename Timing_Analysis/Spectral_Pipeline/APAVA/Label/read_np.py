import numpy as np

arr = np.load("label.npy", allow_pickle=True)

# If it's 1D, make it a column
if arr.ndim == 1:
    arr = arr.reshape(-1, 1)

np.savetxt("file.csv", arr, delimiter=",")