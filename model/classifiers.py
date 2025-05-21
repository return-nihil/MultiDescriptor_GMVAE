import torch
import torch.nn as nn


class Classifier(nn.Module):
    '''Classifier module for supervised clustering of the latent space'''
    def __init__(self, latent_dim, n_classes):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, z):
        return self.net(z)