import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
thisdir = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb
import random

SEED = 1234
np.random.seed(SEED)


def log_spectrograms(spectrogram, reconstruction, tag="MD_GMVAE_spectrograms"):
    '''Log spectrograms to wandb'''
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].imshow(spectrogram[0].squeeze(0).cpu().detach().numpy(), cmap='inferno', origin='lower')
    axs[0].set_title("Original Spectrogram")
    axs[0].axis('off')

    axs[1].imshow(reconstruction[0].squeeze(0).cpu().detach().numpy(), cmap='inferno', origin='lower')
    axs[1].set_title("Reconstructed Spectrogram")
    axs[1].axis('off')
    plt.close(fig)

    wandb.log({tag: wandb.Image(fig)})


def get_class_labels(descriptor, class_mappings='class_mappings.json'):
    '''Get class labels for the given descriptor'''
    with open(os.path.join(thisdir, 'dataset', class_mappings), 'r') as f:
        class_mappings = json.load(f)

    dict_key = f"{descriptor}_dict"
    class_dict = class_mappings[dict_key]

    rev_map = {v: k for k, v in class_dict.items()}
    class_labels = [rev_map[i] for i in sorted(rev_map.keys())]

    return class_labels



def plot_confusion_matrix(cm, descriptor='', output_path='./confusion_matrix'):
    '''Plot and save confusion matrix as PNG and log to wandb'''
    class_labels = get_class_labels(descriptor)
    class_labels = class_labels[:cm.shape[0]]

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion matrix {descriptor}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    os.makedirs(output_path, exist_ok=True)
    png_path = os.path.join(output_path, f"confusion_matrix_{descriptor}.png")
    plt.savefig(png_path)
    plt.close()

    wandb.log({f"Confusion matrix {descriptor}": wandb.Image(png_path)})


def generate_colors(n):
    '''Generate n distinct colors avoiding green hues'''
    exclude_hue_min, exclude_hue_max = 0.22, 0.44  

    allowed_hues = []
    total_range = (exclude_hue_min) + (1 - exclude_hue_max)
    for i in range(n):
        hue_fraction = i / n
        if hue_fraction < exclude_hue_min / total_range:
            h = hue_fraction * total_range
        else:
            h = hue_fraction * total_range + (exclude_hue_max - exclude_hue_min)
        allowed_hues.append(h % 1.0)

    colors = []
    for h in allowed_hues:
        s = 0.95
        v = 0.95
        rgb = hsv_to_rgb([h, s, v])
        colors.append(rgb)

    return colors



def plot_latent_space(latents, true_labels, descriptor, output_path, filename_prefix):
    '''t-SNE and PCA visualizations of the latent space'''

    def plot(latents_2d, method_name):
        full_class_labels = get_class_labels(descriptor)
        unique_labels = np.unique(true_labels)
        filtered_labels = [full_class_labels[i] for i in unique_labels]
        n_classes = len(filtered_labels)
        colors = generate_colors(n_classes)

        fig_width = 8
        legend_width = 3
        fig_height = 8
        sns.set_style("whitegrid")
        plt.rcParams['axes.facecolor'] = (0.85, 0.9, 1, 0.25)
        fig = plt.figure(figsize=(fig_width + legend_width, fig_height))
        ax = fig.add_axes([0, 0, fig_width / (fig_width + legend_width), 1])

        for idx, label in enumerate(filtered_labels):
            class_idx = unique_labels[idx]
            indices = np.where(true_labels == class_idx)[0]
            ax.scatter(latents_2d[indices, 0], latents_2d[indices, 1], label=label,
                       alpha=0.9, s=70, edgecolors='white', color=colors[idx])

        ax.set_title(f"Latent Space - {method_name} - {descriptor}")
        ax.set_xlabel(f'{method_name} dim 1')
        ax.set_ylabel(f'{method_name} dim 2')

        x_min, x_max = latents_2d[:, 0].min(), latents_2d[:, 0].max()
        y_min, y_max = latents_2d[:, 1].min(), latents_2d[:, 1].max()
        max_range = max(x_max - x_min, y_max - y_min)
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2

        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=False)

        os.makedirs(output_path, exist_ok=True)
        file_name = f"{filename_prefix}_{method_name.lower()}_{descriptor}.png"
        save_path = os.path.join(output_path, file_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    latents_tsne = tsne.fit_transform(latents)
    plot(latents_tsne, method_name="tsne")

    # PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    plot(latents_pca, method_name="pca")

