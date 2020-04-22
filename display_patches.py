import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load("patch1000003.npz")
print(npzfile.f.arr_0)

plt.imshow(npzfile.f.arr_0)
plt.show()
