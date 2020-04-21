from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from PIL import Image
import numpy as np

one_image = load_sample_image("china.jpg")

#one_image = Image.open("china.jpg")

#one_image = np.array(one_image)

print('Image shape: {}'.format(one_image.shape))
print('Image type: {}'.format(type(one_image)))

patches = image.extract_patches_2d(one_image, (2, 2))
print('Patches shape: {}'.format(patches.shape))
print('Patches len: {}'.format(len(patches)))

print(patches[1])
print(patches[800])


'''
one_image = Image.open("china.jpg")
print('Image type: {}'.format(type(one_image)))
one_image = np.array(one_image)
print('Image type: {}'.format(type(one_image)))
'''