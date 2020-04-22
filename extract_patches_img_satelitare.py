from sklearn.feature_extraction import image
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import savez_compressed, savez

one_image = Image.open("LC08_L1TP_184029_20160616_20170324_01_T1_B4.tif")
one_image = one_image.resize((2800,2800), Image.NEAREST)
one_image = np.array(one_image)

print('Image shape: {}'.format(one_image.shape))
print('Image type: {}'.format(type(one_image)))

patches = image.extract_patches_2d(one_image, (28, 28))

print('Patches shape: {}'.format(patches.shape))
print('Patches len: {}'.format(len(patches)))

#print(patches[0])
#print(patches[800])

for i in range(len(patches)):
    data = asarray(patches[i])
    savez_compressed('patch'+str(i)+'.npz', data)

