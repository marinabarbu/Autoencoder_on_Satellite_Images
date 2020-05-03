""""https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html"""

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


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir), "Path to videos cannot be found"
        self.images = sorted([os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(".npy")])
        #self.images = sorted([os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(".npy")])[:200]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = np.load(self.images[item])
        #print(img.min(), img.max())
        img = self.transform(img*255).unsqueeze(0)
        return img


def train(train_loader, model, num_epochs=5, lr=1e-4):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            # loss = criterion(recon, data_2_imag)
            loss.backward()
            optimizer.step()
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss.item()))
        outputs.append((epoch, data, recon))
    return outputs

# Antrenarea pe o imagine fara nori din care vrei sa inveti autoencoderul
dataset_train1 = ImageDataset(root_dir='./patches_inainte_alunecare', transform=transforms.ToTensor())
dataset_train1 = list(dataset_train1)

# dataset_train2 = ImageDataset(root_dir='./patches_alunecare', transform=transforms.ToTensor())
# dataset_train2 = list(dataset_train2)

train_loader1 = DataLoader(dataset_train1, batch_size=8, shuffle=True, drop_last=True)
# train_loader2 = DataLoader(dataset_train2, batch_size=8, shuffle=True, drop_last=True)

model = CA.Autoencoder()

max_epochs = 100
outputs1 = train(train_loader1, model, num_epochs=max_epochs)
# outputs2 = train(train_loader2, model, num_epochs=max_epochs)

torch.save(model.state_dict(),"model_compresie")

#print(outputs1)


# for k in range(0, max_epochs, 10):
#     plt.figure(figsize=(9, 2))
#     imgs = outputs[k][1].detach().numpy()
#     recon = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= 9: break
#         plt.subplot(2, 9, i + 1)
#         plt.imshow(item[0])
#
#     for i, item in enumerate(recon):
#         if i >= 9: break
#         plt.subplot(2, 9, 9 + i + 1)
#         plt.imshow(item[0])
#     plt.show()

# testare pe zona cu alunecare I1, I2 300 x 300

# D = np.zeros(shape=I1.shape())
# with torch.no_grad(): #inhiba optimizatorul sa mai invete ceva si sa modifice parametrii
#     for each pixel (u,v):
#         V1 = vecinatate in jurul pixelului [u-14:u+14, v-14:v+14] la momentul T1
#         V2 = vecinatate in jurul pixelului [u-14:u+14, v-14:v+14] la momentul T2
#         y1 = model.codare(V1)
#         y2 = model.codare(V2)
#         d = distanta Euclidiana intre y1 si y2
#         D[u,v] = d
#

'''
with torch.no_grad():
    for data in train_loader:
        recon = model.codare(data)
'''

