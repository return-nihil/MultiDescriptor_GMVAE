import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
  '''2D Convolutional block with optional dropout'''
  def __init__(self,
               in_channels=1,
               out_channels=8,
               kernel_size=3,
               stride=1,
               padding=0,
               dilation=1,
               dropout=0,
               activation=None):
    super(ConvBlock2D, self).__init__()
    activation_functions = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            None: nn.Identity()
            }
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        nn.BatchNorm2d(out_channels),
        activation_functions[activation]
      ]
    layers.insert(2, nn.Dropout(dropout)) if dropout > 0 else None
    self.convblock = nn.Sequential(*layers)

  def forward(self, x):
    x = self.convblock(x)
    return x
     
    
class LatentSpace(nn.Module):
    '''Latent space projection module'''
    def __init__(self, 
                 input_dim=2048, 
                 latent_dim=16, 
                 weigth=1):
        super(LatentSpace, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.weigth = weigth
    
    def _reparametrization_trick(self, mu, logvar, weight):
        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(mu.device) # perche' lo devo mandare a device?
        z = mu + weight * sigma * eps    
        return z

    def forward(self, x):
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = torch.tanh(self._reparametrization_trick(mu, logvar, self.weigth))
        return {'mu': mu, 
                'logvar': logvar, 
                'z': z
                }
    
    
class UpConvBlock2D(nn.Module):
    '''2D Transposed Convolutional block with optional dropout'''
    def __init__(self,
                 in_channels=1,
                 out_channels=8,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dropout=0,
                 activation=None):
        super().__init__()
        activation_functions = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            None: nn.Identity()
        }
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            activation_functions[activation]
        ]
        if dropout > 0:
            layers.insert(2, nn.Dropout2d(dropout))
        self.upblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.upblock(x)
    

class ResBlock2D(nn.Module):
    '''Residual block with optional dropout and activation function'''
    def __init__(
        self,
        channels,
        kernel_size=3,
        padding=1,
        dropout=0,
        activation='silu'
    ):
        super().__init__()
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            None: nn.Identity()
        }
        
        self.act_fn = activation_functions[activation]
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.act_fn(out)
        
        return out