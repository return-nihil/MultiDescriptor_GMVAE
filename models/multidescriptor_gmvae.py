import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
    
from models.blocks import ConvBlock2D, UpConvBlock2D, ResBlock2D, LatentSpace
from utils import get_subconfig


class FeatureExtractor(nn.Module):
    '''Initial feature extractor module'''
    def __init__(self):
        super().__init__()
        config = get_subconfig('feature_extractor')

        self.conv1 = ConvBlock2D(1, **config['conv1'])
        self.conv2a = ConvBlock2D(config['conv1']['out_channels'], **config['conv2a'])
        self.conv2b = ConvBlock2D(config['conv1']['out_channels'], **config['conv2b'])

        in_channels = config['conv2a']['out_channels'] + config['conv2b']['out_channels']
        layers = []
        for layer_cfg in config['post_concat']:
            layers.append(ConvBlock2D(in_channels, **layer_cfg))
            in_channels = layer_cfg['out_channels']
        self.post_concat = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.conv2a(x), self.conv2b(x)], dim=1)
        x = self.post_concat(x)
        return x
    
      
class SubEncoder(nn.Module):
    '''Generic sub-encoder with configurable layers'''
    def __init__(self, in_channels, config):
        super().__init__()
        layers = []
        for params in config:
            layers.append(ConvBlock2D(in_channels, **params))
            in_channels = params['out_channels']
        self.net = nn.Sequential(*layers)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.net(x)
        return self.flat(x)
    
       
class MultidescriptorEncoder(nn.Module):
    '''Modular encoder allowing configurable descriptor sub-encoders'''
    def __init__(self, latent_inputs, latent_dims):
        super().__init__()
        subencoder_configs = get_subconfig('subencoders')

        self.feature_extractor = FeatureExtractor()
        self.sub_encoders = nn.ModuleDict({
            name: SubEncoder(in_channels=self.feature_extractor.output_dim, 
                                       config=config
                                       )
            for name, config in subencoder_configs.items()
        })
        self.latent_spaces = nn.ModuleDict({
            name: LatentSpace(latent_inputs[name], latent_dims[name])
            for name in latent_dims
        })

    def forward(self, x):
        h = self.feature_extractor(x)
        encoded_features = {name: encoder(h) for name, encoder in self.sub_encoders.items()}
        latents = {name: self.latent_spaces[name](encoded_features[name]) for name in encoded_features}
        return latents
    

class Decoder(nn.Module):
    '''Configurable decoder using UpConvBlock2D'''
    def __init__(self, latent_dim):
        super().__init__()
        config = get_subconfig('decoder')

        self.fully = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, 8192),
            nn.BatchNorm1d(8192),
            nn.SiLU()
        )
        
        in_channels = 512
        layers = []
        for layer_cfg in config['upconvs']:
            out_channels = layer_cfg['out_channels']
            upconv_params = {k: v for k, v in layer_cfg.items() if k != 'residual_blocks'}

            layers.append(UpConvBlock2D(in_channels, **upconv_params))

            if 'residual_blocks' in layer_cfg and layer_cfg['residual_blocks'] > 0:
                for _ in range(layer_cfg['residual_blocks']):
                    layers.append(ResBlock2D(
                        channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        dropout=layer_cfg.get('dropout', 0),
                        activation=layer_cfg['activation']
                    ))
            in_channels = out_channels

        self.upblocks = nn.Sequential(*layers)
        
    def forward(self, z):
        x = self.fully(z)
        x = einops.rearrange(x, 'b (c h w) -> b c h w', c=512, h=4, w=4)
        return self.upblocks(x)


class MultiDescriptorGMVAE(nn.Module):
    '''Multi-Descriptor Gaussian Mixture Variational Autoencoder (GMVAE)'''
    def __init__(self, 
                 num_classes
                 ):
        super(MultiDescriptorGMVAE, self).__init__()
        config = get_subconfig('multidescriptor_gmvae')
        latents_dim_config = get_subconfig('latent_dims')
        latents_input_config = get_subconfig('latent_inputs')
        
        self.encoder = MultidescriptorEncoder(latents_input_config, latents_dim_config)
        self.decoder = Decoder(sum(latents_dim_config.values()))
        
        self.mu_lookup = nn.ModuleDict(
            {descriptor: self._build_lookup(num_classes[descriptor], 
                                      latents_dim_config[descriptor], 
                                      trainable=True
                                      ) for descriptor in latents_dim_config
                                    })
        self.logvar_lookup = nn.ModuleDict(
            {descriptor: self._build_lookup(num_classes[descriptor], 
                                      latents_dim_config[descriptor], 
                                      trainable=config['logvar_trainable'], 
                                      pow_exp=config['logvar_init'][descriptor]
                                      ) for descriptor in latents_dim_config
                                    })
        
        self.n_classes = num_classes
     
     
    def _build_lookup(self, num_embeddings, embedding_dim, trainable=False, pow_exp=None):
        '''Build a lookup table for the Gaussian Mixture'''
        lookup = nn.Embedding(num_embeddings, embedding_dim)

        if pow_exp is not None:
            init_logvar = np.log(np.exp(pow_exp)**2)
            nn.init.constant_(lookup.weight, np.log(np.exp(init_logvar) ** 2))
        else:
            nn.init.xavier_uniform_(lookup.weight)
        lookup.weight.requires_grad = trainable

        return lookup
      

    def _log_gauss(self, z, mu, logvar):
        squared_diff = torch.pow(z - mu, 2)
        normalized_squared_diff = squared_diff / torch.exp(logvar)

        log_prob = -0.5 * (
            torch.sum(normalized_squared_diff + logvar, dim=1) + 
            z.size(1) * np.log(2 * np.pi)
        )
        return log_prob
    

    def _approx_class_probabilities(self, z, mu_lookup, logvar_lookup, num_classes):
        '''Estimate class probabilities for a latent vector using a mixture of gaussians'''
        
        class_logits = torch.zeros(z.size(0), num_classes, device=z.device)
        log_prior = np.log(1 / num_classes)
        
        for class_idx in range(num_classes):
            class_idx_tensor = torch.tensor(class_idx, device=z.device)
            class_mean = mu_lookup(class_idx_tensor)
            class_logvar = logvar_lookup(class_idx_tensor)
            class_logits[:, class_idx] = self._log_gauss(z, class_mean, class_logvar) + log_prior
        
        class_prob = torch.nn.functional.softmax(class_logits, dim=1)
        
        return class_logits, class_prob


    def _infer_class(self, z, mu_lookup, logvar_lookup, num_classes):
        '''Predict the most likely class for a given latent vector using the GMM'''

        class_logits, class_prob = self._approx_class_probabilities(
            z, mu_lookup, logvar_lookup, num_classes
        )
        _, predicted_classes = torch.max(class_prob, dim=1)
        
        return class_logits, class_prob, predicted_classes


    def _infer_descriptor_class(self, z, descriptor):
        '''Predict the most likely class for a specific descriptor'''
        mu_lookup = self.mu_lookup[descriptor]
        logvar_lookup = self.logvar_lookup[descriptor]
        num_classes = self.n_classes[descriptor]
        
        return self._infer_class(z, mu_lookup, logvar_lookup, num_classes)
 

    def forward(self, x):
        latents = self.encoder(x)
        z_tot = torch.cat([latents[name]['z'] for name in latents], dim=1)
        output = self.decoder(z_tot)
        return output, latents

    
if __name__ == '__main__':
    num_classes = {
    'timbre': 10,
    'pitch': 12,
    'velocity': 3,
    'duration': 10
    }
    model = MultiDescriptorGMVAE(num_classes)
    model.eval()
    input = torch.randn(1, 1, 256, 256)
    output = model(input)
    print(output[0].shape)
