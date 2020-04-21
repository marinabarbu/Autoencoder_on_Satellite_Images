
# save numpy array as npz file
from numpy import asarray
from numpy import savez_compressed, savez
import numpy as np

# define data
data = asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# save to npy file
savez_compressed('data.npz', data)
b = np.load('data.npz')
print(b.f.arr_0)


data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
savez('data.npz', data)
b = np.load('data.npz')
print(b.f.arr_0)

