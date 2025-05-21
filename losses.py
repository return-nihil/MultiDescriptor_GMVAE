import numpy as np
import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(x_predict, x, reduction="none"):
    loss = F.mse_loss(x_predict, x, reduction=reduction)
    if len(loss.size()) > 2:
        loss = torch.sum(loss, dim=-1)
    return torch.sum(loss, dim=1)


def huber_loss(x_predict, x, reduction="none"):
    loss = F.smooth_l1_loss(x_predict, x, reduction=reduction)
    if len(loss.size()) > 2:
        loss = torch.sum(loss, dim=-1)
    return torch.sum(loss, dim=1)


def bce_loss(x_predict, x, reduction="none"):
    loss = F.binary_cross_entropy_with_logits(x_predict, x, reduction=reduction)
    return torch.sum(loss, dim=1)


def ce_loss(x_predict, x, reduction="none"):#, label_idx=None):
    loss = F.cross_entropy(x_predict, x, reduction=reduction)
    return loss


def kl_gauss(q_mu, q_logvar, mu=None, logvar=None):
    """
    KL divergence between two diagonal gaussians
    """
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    return -0.5 * (1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar))


def kl_class(log_q_y_logit, q_y, k=10):
    q_y_shape = list(q_y.size())

    if not q_y_shape[1] == k:
        raise ValueError("q_y_shape (%s) does not match the given k (%s)" % (
            q_y_shape, k))

    h_y = torch.sum(q_y * torch.nn.functional.log_softmax(log_q_y_logit, dim=1), dim=1)

    return h_y - np.log(1 / k), h_y


def kl_latent(q_mu, q_logvar, q_y, mu_lookup, logvar_lookup):
    """
    q_z (b, z)
    q_y (b, k)
    mu_lookup (k, z)
    logvar_lookup (k, z)
    """
    mu_lookup_shape = [mu_lookup.num_embeddings, mu_lookup.embedding_dim]  # (k, z_dim)
    logvar_lookup_shape = [logvar_lookup.num_embeddings, logvar_lookup.embedding_dim]  # (k, z_dim)
    q_mu_shape = list(q_mu.size())
    q_logvar_shape = list(q_logvar.size())
    q_y_shape = list(q_y.size())

    if not np.all(mu_lookup_shape == logvar_lookup_shape):
        raise ValueError("mu_lookup_shape (%s) and logvar_lookup_shape (%s) do not match" % (
            mu_lookup_shape, logvar_lookup_shape))
    if not np.all(q_mu_shape == q_logvar_shape):
        raise ValueError("q_mu_shape (%s) and q_logvar_shape (%s) do not match" % (
            q_mu_shape, q_logvar_shape))
    if not q_y_shape[0] == q_mu_shape[0]:
        raise ValueError("q_y_shape (%s) and q_mu_shape (%s) do not match in batch size" % (
            q_y_shape, q_mu_shape))
    if not q_y_shape[1] == mu_lookup_shape[0]:
        raise ValueError("q_y_shape (%s) and mu_lookup_shape (%s) do not match in number of class" % (
            q_y_shape, mu_lookup_shape))

    batch_size, n_class = q_y_shape
    kl_sum = torch.zeros(batch_size, n_class).to(q_mu.device)  # create place holder

    for k_i in torch.arange(0, n_class):
        #print(k_i.device)
        k_i = k_i.to(q_mu.device)
        kl_sum[:, k_i] = torch.sum(kl_gauss(q_mu, q_logvar, mu_lookup(k_i), logvar_lookup(k_i)), dim=1)
        kl_sum[:, k_i] *= q_y[:, k_i]

    return torch.sum(kl_sum, dim=1)  # sum over classes

def kl_emb(mu_lookup, logvar_lookup, mu, logvar, y):
    return torch.sum(kl_gauss(mu, logvar,
                              mu_lookup(y), logvar_lookup(y)), dim=1)
    

########################################################################################################################
def classifier_loss(target_classes, logits, weights=None):
    tot_loss = 0
    losses = {}
    if weights is None:
        weights = {class_name: 1 for class_name in target_classes}
    for class_name in target_classes:
        losses[class_name] = ce_loss(logits[class_name], target_classes[class_name])
        tot_loss += weights[class_name] * losses[class_name]
    return tot_loss, losses

def descriptor_loss(model, latents, target_classes, descriptors_names):
    # Assert that all latents, target_classes and descriptors_names have the same keys. Otherwise, raise an error
    assert set(latents.keys()) == set(target_classes.keys()) == set(descriptors_names), "Keys mismatch. Check the keys of latents, target_classes and descriptors_names"
    loss = 0
    for name in descriptors_names:
        loss += kl_emb(model.mu_lookup[name], model.logvar_lookup[name], latents[name][0], latents[name][1], target_classes[name])
    return loss

        


# Ardan Code ############################################################################################################
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def remover_kl_uniform_loss(model, latent, number_of_targets, lambda_adv=0.5):
    # Get predicted probabilities from the model
    predicted_probs = model(latent)  
    
    # Ensure predicted_probs is non-negative and clamp any small values to avoid log(0)
    predicted_probs = torch.clamp(predicted_probs, min=1e-8)

    # Create uniform probability distribution
    uniform_probs = torch.full_like(predicted_probs, 1.0/ number_of_targets).to(latent.device)#) #
    
    # Check that the uniform distribution sums to 1
    #uniform_probs = uniform_probs / uniform_probs.sum(dim=-1, keepdim=True)
    
    # Compute the KL divergence (log probabilities for predicted)
    kl_div = F.kl_div(predicted_probs.log(), uniform_probs, reduction='batchmean')
    
    # Final loss with the scaling factor lambda_adv
    loss = lambda_adv * kl_div
    
    return loss


def remover_loss(remover, zs, target):
    y = remover(zs)
    loss = F.cross_entropy(y, target)
    return loss


def weighted_mse_loss(recon, target):
        # Ensure inputs are of the same shape
        if recon.shape != target.shape:
            raise ValueError(f"Shape mismatch: recon {recon.shape}, target {target.shape}")
        
        # Compute gradients of the target
        grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # Vertical gradient
        grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # Horizontal gradient

        # Combine gradients (align shapes)
        grad_y = grad_y[:, :, :, :-1]  # Remove last column for alignment
        grad_x = grad_x[:, :, :-1, :]  # Remove last row for alignment
        gradient_magnitude = grad_x + grad_y

        # Pad the gradient to match the input size
        weight_map = torch.nn.functional.pad(gradient_magnitude, (0, 1, 0, 1))

        # Compute weighted MSE
        weighted_mse = torch.mean(weight_map * (recon - target) ** 2)
        return weighted_mse
    
        
