import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
    
from blocks import ConvBlock2D, UpConvBlock2D, ResBlock2D, LatentSpace


feature_extractor_config = {
    'conv1': {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'activation': 'leaky_relu'},
    'conv2a': {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'activation': 'leaky_relu'},
    'conv2b': {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 2, 'dilation': 2, 'dropout': 0.1, 'activation': 'leaky_relu'},
    'post_concat': [
        {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dropout': 0.1, 'activation': 'leaky_relu'},
        {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'tanh'}
    ]
}

subencoder_configs = {
    'timbre': [
        {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'activation': 'leaky_relu', 'dropout': 0.2},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'activation': 'silu'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'}
    ],
    'pitch': [
        {'out_channels': 256, 'kernel_size': [2,5], 'stride': [2,4], 'padding': [0,1], 'activation': 'relu', 'dropout': 0.2},
        {'out_channels': 128, 'kernel_size': [2,4], 'stride': [2,4], 'padding': [0,1], 'activation': 'silu'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'}
    ],
    'velocity': [
        {'out_channels': 256, 'kernel_size': [5,2], 'stride': [4,2], 'padding': [1,0], 'activation': 'relu', 'dropout': 0.2},
        {'out_channels': 128, 'kernel_size': [4,2], 'stride': [4,2], 'padding': [1,0], 'activation': 'silu'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'}
    ],
    'duration': [
        {'out_channels': 256, 'kernel_size': [5,2], 'stride': [4,2], 'padding': [1,0], 'activation': 'relu', 'dropout': 0.2},
        {'out_channels': 128, 'kernel_size': [4,2], 'stride': [4,2], 'padding': [1,0], 'activation': 'silu'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'silu'}
    ]
}

enhanced_decoder_config = {
    'fc1': 512,
    'fc2': 8192,
    'reshape': {'channels': 512, 'height': 4, 'width': 4},
    'upconvs': [
        {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'silu', 'residual_blocks': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'silu', 'residual_blocks': 0},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'silu', 'residual_blocks': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'tanh', 'residual_blocks': 0},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'silu', 'residual_blocks': 1},
        {'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'activation': 'silu', 'residual_blocks': 0},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0, 'activation': None, 'residual_blocks': 0}
    ]
}







latent_inputs = {
    'timbre': 8192,
    'pitch': 2048,
    'velocity': 2048,
    'duration': 2048
}

latent_dims = {
    'timbre': 8,
    'pitch': 4,
    'velocity': 4,
    'duration': 4
}

latent_classes = {
    'timbre': 14,
    'pitch': 61,
    'velocity': 3,
    'duration': 24
}



class FeatureExtractor(nn.Module):
    '''Initial feature extractor module'''
    def __init__(self, config):
        super().__init__()

        self.conv1 = ConvBlock2D(1, **config['conv1'])
        self.conv2a = ConvBlock2D(config['conv1']['out_channels'], **config['conv2a'])
        self.conv2b = ConvBlock2D(config['conv1']['out_channels'], **config['conv2b'])

        in_channels = config['conv2a']['out_channels'] + config['conv2b']['out_channels']
        layers = []
        for layer_cfg in config['post_concat']:
            layers.append(ConvBlock2D(in_channels, **layer_cfg))
            in_channels = layer_cfg['out_channels']
        self.post_concat = nn.Sequential(*layers)

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
    def __init__(self, subencoder_configs, latent_inputs, latent_dims):
        super().__init__()
        self.feature_extractor = FeatureExtractor() # Initial feature extractor module

        # Create sub-encoders for each descriptor in the config
        self.sub_encoders = nn.ModuleDict({
            name: SubEncoder(in_channels=self.feature_extractor.output_dim, 
                                       config=config
                                       )
            for name, config in subencoder_configs.items()
        })

        # Create latent spaces for each descriptor in the config
        self.latent_spaces = nn.ModuleDict({
            name: LatentSpace(latent_inputs[name], latent_dims[name])
            for name in latent_dims
        })

    def forward(self, x):
        # Extract feature vector h from the input spectrogram
        h = self.feature_extractor(x)
        # Encode feature vector h into latent spaces
        encoded_features = {name: encoder(h) for name, encoder in self.sub_encoders.items()}
        latents = {name: self.latent_spaces[name](encoded_features[name]) for name in encoded_features}
        return latents
    

class Decoder(nn.Module):
    '''Configurable decoder using UpConvBlock2D'''
    def __init__(self, latent_dim, config):
        super().__init__()
        self.fully = nn.Sequential(
            nn.Linear(latent_dim, config['fc1']),
            nn.BatchNorm1d(config['fc1']),
            nn.SiLU(),
            nn.Linear(config['fc1'], config['fc2']),
            nn.BatchNorm1d(config['fc2']),
            nn.SiLU()
        )
        self.reshape_channels = config['reshape']['channels']
        self.reshape_height = config['reshape']['height']
        self.reshape_width = config['reshape']['width']
        
        in_channels = self.reshape_channels
        layers = []
        
        for layer_cfg in config['upconvs']:
            out_channels = layer_cfg['out_channels']
            layers.append(UpConvBlock2D(in_channels, **layer_cfg))
            
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
        x = einops.rearrange(x, 'b (c h w) -> b c h w', 
                             c=self.reshape_channels,
                             h=self.reshape_height,
                             w=self.reshape_width)
        return self.upblocks(x)




class MultidescriptorGMVAE(nn.Module):
    def __init__(self, 
                 subencoder_configs,
                 latent_inputs,
                 latent_dims,
                 latent_classes,
                 logvar_init={'timbre': 0.0, 'pitch': -2.0, 'velocity': -2.0, 'duration': -2.0},
                 logvar_trainable=False,
                 ):
        super(MultidescriptorGMVAE, self).__init__()
        
        # Assert that subencoder_configs, latent_inputs, latent_dims, and latent_classes have the same keys
        assert set(subencoder_configs.keys()) == set(latent_inputs.keys()) == set(latent_dims.keys()) == set(latent_classes.keys()) == set(logvar_init.keys()), \
        "Keys in subencoder_configs, latent_inputs, latent_dims, and latent_classes must match.\n Got:\n subencoder_config: {}\n, latent_inputs: {}\n, latent_dims: {}\n, latent_classes: {}\n, logvar_init: {}".format(subencoder_configs.keys(), latent_inputs.keys(), latent_dims.keys(), latent_classes.keys(), logvar_init.keys())
        
        # Set device
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Set configurations
        self.latent_dims = latent_dims
        self.subencoder_configs = subencoder_configs
        self.latent_classes = latent_classes
        
        # Create the encoder and decoder
        self.encoder = MultidescriptorEncoder(subencoder_configs, latent_inputs, latent_dims)
        self.decoder = Decoder(sum(latent_dims.values()))
        
        # Create mu and logvar lookups for the gaussian mixture
        self.mu_lookup = nn.ModuleDict({name: self._build_lookup(latent_classes[name], latent_dims[name], trainable=True) for name in latent_dims})
        self.logvar_lookup = nn.ModuleDict({name: self._build_lookup(latent_classes[name], latent_dims[name], trainable=logvar_trainable, pow_exp=logvar_init[name]) for name in latent_dims})
     
        
    def _build_lookup(self, num_embeddings, embedding_dim, trainable=False, pow_exp=None):
        '''Build a lookup table for the Gaussian Mixture'''
        #lookup = nn.Embedding(num_embeddings, embedding_dim).to(self.device)
        lookup = nn.Embedding(num_embeddings, embedding_dim)
        if pow_exp is not None:
            init_sigma = np.exp(pow_exp)
            init_logvar = np.log(init_sigma ** 2)
            nn.init.constant_(lookup.weight, np.log(np.exp(init_logvar) ** 2))
        else:
            nn.init.xavier_uniform_(lookup.weight)
        lookup.weight.requires_grad = trainable
        return lookup
     
      
    def _log_gauss(self, q_z, mu, logvar):
        return torch.sum(-0.5 * (torch.pow(q_z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi)), dim=1)
    
    
    def _approx_q_y(self, q_z, mu_lookup, logvar_lookup, n_class):
        '''Approximate q_y. For each class k, compute the log-likelihood of q_z under the Gaussian Mixture'''
        batch_size = q_z.size(0)
        #log_q_y_logit = torch.zeros(batch_size, n_class, device=self.device)
        log_q_y_logit = torch.zeros(batch_size, n_class, device=q_z.device)
        for k in range(n_class):
            # mu_k, logvar_k = mu_lookup(torch.tensor(k, device=self.device)), logvar_lookup(torch.tensor(k, device=self.device))
            mu_k, logvar_k = mu_lookup(torch.tensor(k, device=q_z.device)), logvar_lookup(torch.tensor(k, device=q_z.device))
            log_q_y_logit[:, k] = self._log_gauss(q_z, mu_k, logvar_k) + np.log(1 / n_class)
        return log_q_y_logit, torch.nn.functional.softmax(log_q_y_logit, dim=1)
    
    
    def _infer_class(self, z, mu_lookup, logvar_lookup, n_class):
        '''Infer the most likely class for a given latent vector q_z using the Gaussian Mixture'''
        log_q_y_logit, q_y = self._approx_q_y(z, mu_lookup, logvar_lookup, n_class)
        val, ind = torch.max(q_y, dim=1)
        return log_q_y_logit, q_y, ind
    
    
    def infer_descriptor_class(self, q_z, descriptor='timbre'):
        '''Infer the most likely class of q_z for a given descriptor'''
        assert descriptor in list(self.latent_dims.keys()), f"Descriptor '{descriptor}' not found in the model. Available descriptors: {list(self.latent_dims.keys())}"
        return self._infer_class(q_z, self.mu_lookup[descriptor], self.logvar_lookup[descriptor], self.latent_classes[descriptor])
        
        
    def forward(self, x):
        # Encode input spectrogram x into the various latent spaces (timbre, pitch, velocity, duration)
        latents = self.encoder(x)
        # Concatenate the latent vectors from each descriptor into a single latent vector
        # z_timbre = latents['timbre'][-1]
        # z_pitch = latents['pitch'][-1]
        # z_velocity = latents['velocity'][-1]
        # z_duration = latents['duration'][-1]
        # z_tot = torch.cat([z_timbre, z_pitch, z_velocity, z_duration], dim=1)
        z_tot = torch.cat([latents[name][-1] for name in latents], dim=1)
        # Decode the concatenated latent vector into the output spectrogram
        return self.decoder(z_tot), latents

    
if __name__ == '__main__':
    import json 
    
    with open('configs/multidescriptor_default.json', 'r') as f:
        config = json.load(f)
    subencoder_configs = config['subencoder_configs']
    latent_inputs = config['latent_inputs']
    latent_dims = config['latent_dims']
    latent_classes = config['latent_classes']
      
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model = MultidescriptorGMVAE(latent_inputs=latent_inputs, latent_dims=latent_dims, subencoder_configs=subencoder_configs, latent_classes=latent_classes).to(device)
    model.eval()
    print("Created MultidescriptorGMVAE: \n", model)
    
    # Display the number of parameters
    number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[INFO] Number of parameters", number_of_params)

    # Test the forward pass with dummy data
    input = torch.randn(1, 1, 256, 256).to(device)
    output, latents = model(input)
    print("Output Spectrogram: ", output.size())
    print('>'*20,"LATENTS", '<'*20, sep='\n')
    for name, latent in latents.items():
        print(f"{name.capitalize()} Mu", latent[0].size())
        print(f"{name.capitalize()} Logvar", latent[1].size())
        print(f"{name.capitalize()} Z", latent[2].size())
        print("-"*20)
     
    # Infer the class of the latent vector for the timbre descriptor   
    descriptor = 'timbre'    
    y_k = model.infer_descriptor_class(latents[descriptor][2], descriptor=descriptor)[-1]
    print("Inferred Class", y_k)
    
    print('='*10,"Mu lookups", '='*10, sep='\n')
    # Print the model's lookup tables
    for name, lookup in model.mu_lookup.items():
        print(name.capitalize(), lookup.weight.size())
        print("-"*20)
        
    print('='*10,"Logvar lookups", '='*10, sep='\n')
    for name, lookup in model.logvar_lookup.items():
        print(name.capitalize(), lookup.weight.size())
        print("-"*20)
        
    # import json
    # # Save configs to a json file
    # with open('configs/multidescriptor_default.json', 'w') as f:
    #     json.dump({'subencoder_configs': subencoder_configs, 'latent_inputs': latent_inputs, 'latent_dims': latent_dims, 'latent_classes': latent_classes}, f, indent=4)