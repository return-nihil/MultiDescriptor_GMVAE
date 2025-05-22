import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_subconfig


class Reconstruction_Loss(nn.Module):
    '''Weighted MSE loss + Huber loss'''
    def __init__(self, recon_lambda=get_subconfig('losses').get('recon_lambda'),
                 huber_weight=1.0, 
                 mse_weight=1.0):
        super().__init__()
        self.recon_lambda = recon_lambda
        self.huber_weight = huber_weight
        self.mse_weight = mse_weight

    def forward(self, recon, target):

        grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]) 
        grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]) 
        grad_y = grad_y[:, :, :, :-1]
        grad_x = grad_x[:, :, :-1, :]
        gradient_magnitude = grad_x + grad_y
        weight_map = F.pad(gradient_magnitude, (0, 1, 0, 1))
        weighted_mse = torch.mean(weight_map * (recon - target) ** 2)

        huber = F.smooth_l1_loss(recon, target, reduction='none').mean()

        total_loss = self.mse_weight * weighted_mse + self.huber_weight * huber
        return total_loss * self.recon_lambda
    

class KL_Emb_Loss(nn.Module):
    '''KL divergence loss for embeddings'''
    def __init__(self, 
                 kl_lambda=get_subconfig('losses').get('kl_lambda')):
        super().__init__()
        self.kl_lambda = kl_lambda

    def forward(self, mu_lookup, logvar_lookup, mu, logvar, y):
        prior_mu = mu_lookup(y)          
        prior_logvar = logvar_lookup(y) 

        kl = 0.5 * (
            prior_logvar - logvar +
            (torch.exp(logvar) + (mu - prior_mu) ** 2) / torch.exp(prior_logvar) - 1
        )
        return torch.sum(kl, dim=1) * self.kl_lambda


class Classifier_Loss(nn.Module):
    '''Cross-entropy loss for classification'''
    def __init__(self,
                 class_lambda=get_subconfig('losses').get('class_lambda')):
        super().__init__()
        self.class_lambda = class_lambda

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets) * self.class_lambda
    

class Remover_Loss(nn.Module):
    '''Cross-entropy loss for removers'''
    def __init__(self,
                 rem_lambda=get_subconfig('losses').get('rem_lambda')):
        super().__init__()
        self.rem_lambda = rem_lambda

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets) * self.rem_lambda
    

class Remover_KL_Uniform_Loss(nn.Module):
    '''KL divergence loss for removers' uniform distribution'''
    def __init__(self, 
                 lambda_adv=get_subconfig('losses').get('rem_kl_lambda')):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, logits, number_of_targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-8)

        uniform = torch.full_like(probs, 1.0 / number_of_targets)
        kl_div = F.kl_div(probs.log(), uniform, reduction='batchmean')

        return self.lambda_adv * kl_div