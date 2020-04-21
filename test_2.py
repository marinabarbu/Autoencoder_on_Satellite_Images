from cae import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import urllib, gzip, pickle

mnistfile = 'mnist.pkl.gz'
if not os.path.isfile(mnistfile):
    url = urllib.request.URLopener()
    url.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", mnistfile)
f = gzip.open(mnistfile, 'rb')
training_set, validation_set, testing_set = pickle.load(f, encoding='latin1')

f.close()

print(type(training_set)) #tuplu
print(len(training_set)) # 2

X_train, y_train = training_set
print(type(X_train))
print(len(X_train)) #50000 de imagini
print(type(X_train[0])) # numpy.ndarray
#print(X_train[0])# vector - imagine cred

print(X_train[0].shape) #(784, )
X_validate, y_validate = validation_set
X_test, y_test = testing_set

def reshape_mnist(vectors):
    images = np.reshape(vectors, (-1,28,28,1))
    return images

X_train = reshape_mnist(X_train)
X_validate = reshape_mnist(X_validate)
X_test = reshape_mnist(X_test)

print(X_train[0].shape)

img1 = Image.open("LC08_L1TP_184029_20160616_20170324_01_T1_B4.tif")
print(type(img1)) #PIL TIFF
#img1 = np.reshape(img1)
img1 = np.array(img1) 
print(type(img1)) # numpy.ndarray
print(type(img1[0][0]))
img1 = img1.astype(float)
print(type(img1[0][0]))
print(len(img1)) # 7751
print(img1.shape) # (7751, 7631)
img1 = np.resize(img1,(28,28)) #(28, 28)
print(img1.shape)
img1 = img1.flatten() 
img1 = np.reshape(img1, (-1,28,28,1))
print(img1.shape) # (59147881, ) #dupa resize (784, )
#print(img1)

#print(imgarray1.shape)
img2 = Image.open("LC08_L1TP_184029_20170603_20170615_01_T1_B4.tif")
img2 = np.array(img2)
img2 = img2.astype(float)
img2 = np.resize(img2,(28,28))
img2 = img2.flatten() 

img2 = np.reshape(img2, (-1,28,28,1))
#img2 = reshape_mnist(img2)

images =np.array([img1,img2])

print(X_train[0].shape)

#img1 = np.reshape(img1, X_train[0].shape)
#img2 = np.reshape(img2, X_train[0].shape)

def plot_many(images, n=[1,2]):
    plt.figure(figsize=(2,1))
    for i in range(n[0]):
        for j in range(n[1]):
            plt.subplot(n[0], n[1], j*n[0]+i+1)
            plt.imshow(images[np.random.randint(0, images.shape[0]-1),:,:,0], cmap=plt.get_cmap("Greys"))
            plt.axis('off')
      
#plt.plot(img1)
#plt.imshow(img1[np.random.randint(0, img1.shape[0]-1),:,:,0],cmap=plt.get_cmap("Greys"))
#plt.axis('off')


#plot_many(images)

#plt.plot(img2)

print(X_train.shape[1])
print(X_train.shape[2])

cae = ConvAutoEncoder(input_shape=(28,28,1), output_dim=2, filters=[32,64])

# fit to training, validation data
cae.fit(img1, img2, epochs=20)
cae.save_weights()


print("Validation")
print(X_validate[0].shape)
print(X_train[0].shape)

print(img1.shape)
print(img2.shape)


'''
plot_many(X_train)



# initialize ConvAutoEncoder model
cae = ConvAutoEncoder(input_shape=(X_train.shape[1], X_train.shape[2],1),
                      output_dim=10, filters=[32,64])



# fit to training, validation data
cae.fit(X_train, X_validate, epochs=20)
cae.save_weights()

# predict on testing data
test_codes = cae.encode(X_test)

# run tSNE on testing codes
from sklearn.manifold import TSNE
test_codes_embedded = TSNE(n_components=2).fit_transform(test_codes)

# plot tSNE results with MNIST labels
fig, ax = plt.subplots()
plt.scatter(test_codes_embedded[:,0], test_codes_embedded[:,1], c=y_test, cmap=plt.get_cmap("tab10"))
plt.colorbar()

def plot_many_vs_reconstructed(original, reconstructed, n=10):
    plt.figure(figsize=(12,4))
    n = [2,n]
    for i in range(n[1]):
        idx = np.random.randint(0, original.shape[0]-1)
        plt.subplot(n[0], n[1], i+1)
        plt.imshow(original[idx,:,:,0], cmap=plt.get_cmap("Greys"))
        plt.axis('off')
        plt.subplot(n[0], n[1], i+1+n[1])
        plt.imshow(reconstructed[idx,:,:,0], cmap=plt.get_cmap("Greys"))
        plt.axis('off')
        
# compare original and reconstructed images
test_reconstructed = cae.decode(test_codes)

plot_many_vs_reconstructed(X_test, test_reconstructed)
'''