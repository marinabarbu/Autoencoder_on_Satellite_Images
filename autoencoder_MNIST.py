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


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir), "Path to videos cannot be found"
        self.images = sorted([os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(".npz")])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = cv2.imread(self.images[item])
        img_data = self.transform(img).unsqueeze(0)
        return img_data


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.encoder(x)
        xp = self.decoder(y)
        return xp


def train(train_loader, model, num_epochs=5, lr=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        outputs.append((epoch, img, recon), )
    return outputs


mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:1000]
train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=32, shuffle=True)


model = Autoencoder()
max_epochs = 20
outputs = train(train_loader, model, num_epochs=max_epochs)
for k in range(0, max_epochs, 5):
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i + 1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0])
    plt.show()
