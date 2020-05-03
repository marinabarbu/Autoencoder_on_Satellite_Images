import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import clasa_autoencoder as CA
from autoencoder_img_satelitare2 import ImageDataset, train
from scipy.spatial import distance
import csv
from math import sqrt
import pickle


dataset_train1 = ImageDataset(root_dir='./patches_img300x300_inainte', transform=transforms.ToTensor())
dataset_train1 = list(dataset_train1)

dataset_train2 = ImageDataset(root_dir='./patches_img300x300_dupa', transform=transforms.ToTensor())
dataset_train2 = list(dataset_train2)

train_loader1 = DataLoader(dataset_train1, batch_size=8, shuffle=True, drop_last=True)
train_loader2 = DataLoader(dataset_train2, batch_size=8, shuffle=True, drop_last=True)

model = CA.Autoencoder()
model.load_state_dict(torch.load("model_compresie"))
# model.eval()

F1 = []
F2 = []
with torch.no_grad():
    for data in train_loader1:
        features1 = model.codare(data) # torch tensor de dimensiune 32 X 64 X 7
        F1.append(features1.detach().numpy().flatten()) # o lista de 90000/4 vectori numpy de dimensiune (14336, )
    for data in train_loader2:
        features2 = model.codare(data)
        F2.append(features2.detach().numpy().flatten())


with open("F1", 'wb') as f:
    pickle.dump(F1, f)

with open("F2", 'wb') as f:
    pickle.dump(F2, f)

dist_euclid_list = []


for i in range(len(F1)):
    dist = 0.0
    for j in range(len(F1[i])):
        dist += abs(F1[i][j] - F2[i][j])
    dist_euclid_list.append(dist)



#dist_euclid = distance.euclidean(F1, F2)
with open('dist_euclid_file.csv', 'w', newline='') as dist_euclid_file:
    writer = csv.writer(dist_euclid_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in dist_euclid_list:
        writer.writerow(row)


