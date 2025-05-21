import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import datetime

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


from losses import *
from utils import EarlyStopping
from models.classifiers import Classifier, Remover
from models.multidescriptor_gmvae import MultiDescriptorGMVAE
from dataset.dataset import TinySol_Dataset

#from src.utils.utils import plot_confusion_matrix

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import get_subconfig, xavier_init, kaiming_init

# Get the current date and time for the wandb run name and the model checkpoints
now = datetime.datetime.now()
DT_STRING = now.strftime("%d-%m-%Y_%H-%M-%S")

# PATHS
CHECKPOINT_PATH = f'runs/{DT_STRING}/checkpoints'
MODELPATH = f'runs/{DT_STRING}/checkpoints/multidescriptor_GMVAE_{DT_STRING}.pth'
CLASSIFIER_BESTLOSS = f'runs/{DT_STRING}/checkpoints/classifiers_bestloss_{DT_STRING}.pth'
CLASSIFIER_PATH = f'runs/{DT_STRING}/checkpoints/classifiers_{DT_STRING}.pth'

# CONFUSION MATRIX
CONFUSION_PATH = f'runs/{DT_STRING}/confusion_matrix'
DATAFRAME = 'dataset/dataframes/tinysol_dataframe_numpy.csv'

data_config = get_subconfig('data')
train_config = get_subconfig('train')
device = get_subconfig('device')



# LOGS
LOGS = True
METRICS = True
PROJECT_NAME = 'MD-VAE_TINYSOL'

# Seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


# Beta warm-up (epochs)
WARMUP = 250

# Pretraining withouth removers
PRETRAIN = 250

# Weight for the losses
RECON_LAMBDA = 50 #1
CLASS_LAMBDA = 2 #1
REM_LAMBDA = 0.1



      
def train_multidescriptor(model, 
                          dataloaders, 
                          classifiers, 
                          removers, 
                          bs, 
                          num_epochs, 
                          lr, 
                          scheduler_patience, 
                          early_stopping_patience):
    model.apply(xavier_init)
    model.to(device)
    
    if removers is not None:
        for name, remover in removers.items():
            remover.apply(kaiming_init)
            # Take parameters for optimization            
        optimizer_removers = Adam([param for rem in removers.values() for param in rem.parameters()], lr=lr)
        optimizer_removers = Adam(removers.parameters(), lr=lr)
        
    optimizer_model = Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=50, min_lr=1e-6) #ReduceLROnPlateau(optimizer_model, mode='min', factor=0.1, patience=10, min_lr=1e-6)

    best_loss = float('inf')
    best_classifier_loss = float('inf')
    early_stopping = EarlyStopping(patience=1000, min_delta=0.01)

    for epoch in range(num_epochs):
        print('-' * 50)
        print(f'[EPOCH] {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            print(f'Phase: {phase}')

            total_loss = 0
            total_recon = 0
            total_timbre_kld = 0
            total_pitch_kld = 0
            total_velocity_kld = 0
            total_duration_kld = 0
            total_timbre_loss = 0
            total_pitch_loss = 0
            total_velocity_loss = 0
            total_duration_loss = 0
            total_removers_KL_loss = 0
            total_removers_loss = 0
            total_classier_loss = 0

            for S, T, P, V, D in tqdm(dataloaders[phase]):
                spectrogram, timbre_label, pitch_label, velocity_label, duration_label = S.to(device), T.to(device), P.to(device), V.to(device), D.to(device)
                    
                target_classes = {'timbre': timbre_label,
                                    'pitch': pitch_label,
                                    'velocity': velocity_label,
                                    'duration': duration_label
                                    }

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                x_predict, latents = model(spectrogram)
                
                mse = weighted_mse_loss(x_predict, spectrogram)
                huber_loss = F.huber_loss(x_predict, spectrogram)
                recon_loss = torch.mean(mse + huber_loss, dim=0)
                class_logits = {}
                class_probs = {}
                neg_kld_z = {}
                for descriptor in target_classes.keys():
                    class_logits[descriptor], class_probs[descriptor], _ = model._infer_descriptor_class(
                        latents[descriptor]['z'], descriptor=descriptor
                    )
                    neg_kld_z[descriptor] = -1 * kl_latent(
                        latents[descriptor]['mu'], 
                        latents[descriptor]['logvar'], 
                        class_probs[descriptor], 
                        model.mu_lookup[descriptor], 
                        model.logvar_lookup[descriptor]
                    ) * 0.00025

                timbre_kld = torch.mean(neg_kld_z['timbre'], dim=0)
                pitch_kld = torch.mean(neg_kld_z['pitch'], dim=0)
                velocity_kld = torch.mean(neg_kld_z['velocity'], dim=0)
                duration_kld = torch.mean(neg_kld_z['duration'], dim=0)

                class_tot_loss, class_losses = classifier_loss(
                    target_classes, 
                    class_logits, 
                    weights={'timbre': 4, 'pitch': 2, 'velocity': 2, 'duration': 2}
                )
                
                class_tot_loss = torch.mean(class_tot_loss, dim=0) 
                for name, loss in class_losses.items():
                    class_losses[name] = torch.mean(loss, dim=0)  
                 
                beta = epoch/WARMUP
                lower_bound = -1 * ((recon_loss) + (timbre_kld + pitch_kld + velocity_kld + duration_kld) * min(beta, 1.0)) # 
                print(f'[LOWER_BOUND] {lower_bound.item()}')
                print(f'[RECON_LOSS] {recon_loss.item()}')
                print(f'[pitch_KLD] {pitch_kld.item()}')
                overall_loss = lower_bound  + class_tot_loss 

                t_remover_input = torch.cat((latents['pitch']['z'], latents['velocity']['z'], latents['duration']['z']), dim=1)
                p_remover_input = torch.cat((latents['timbre']['z'], latents['velocity']['z'], latents['duration']['z']), dim=1)
                v_remover_input = torch.cat((latents['timbre']['z'], latents['pitch']['z'], latents['duration']['z']), dim=1)
                l_remover_input = torch.cat((latents['timbre']['z'], latents['pitch']['z'], latents['velocity']['z']), dim=1)                        

        # FIRST STAGE
                if removers is not None and epoch >= PRETRAIN:
                    beta_rem = REM_LAMBDA*(epoch-PRETRAIN)/WARMUP
                    for _, remover in removers.items():
                        remover.eval()

                    t_remover_KL_loss = remover_kl_uniform_loss(removers['timbre'], t_remover_input, model.latent_classes['timbre'])
                    p_remover_KL_loss = remover_kl_uniform_loss(removers['pitch'], p_remover_input, model.latent_classes['pitch'])
                    v_remover_KL_loss = remover_kl_uniform_loss(removers['velocity'], v_remover_input, model.latent_classes['velocity'])
                    l_remover_KL_loss = remover_kl_uniform_loss(removers['duration'], l_remover_input, model.latent_classes['duration'])
                    total_removers_KL_loss += t_remover_KL_loss.item() + p_remover_KL_loss.item() + v_remover_KL_loss.item() + l_remover_KL_loss.item()
                
                else:
                    beta_rem = 0
                    t_remover_KL_loss = torch.zeros(1).to(device)
                    p_remover_KL_loss = torch.zeros(1).to(device)
                    v_remover_KL_loss = torch.zeros(1).to(device)
                    l_remover_KL_loss = torch.zeros(1).to(device)

                    total_removers_kl_loss = torch.zeros(1).to(device)

                first_loss = -1 * overall_loss + (t_remover_KL_loss + p_remover_KL_loss + v_remover_KL_loss + l_remover_KL_loss) * min(beta_rem, REM_LAMBDA)

                total_loss += first_loss.item()
                total_recon += recon_loss.item()
                total_timbre_kld += timbre_kld.item()
                total_pitch_kld += pitch_kld.item()
                total_velocity_kld += velocity_kld.item()
                total_duration_kld += duration_kld.item()
                total_timbre_loss += class_losses['timbre'].item()
                total_pitch_loss += class_losses['pitch'].item()
                total_velocity_loss += class_losses['velocity'].item()
                total_duration_loss += class_losses['duration'].item()
                total_classier_loss += class_tot_loss.item()

                if phase == 'train':
                    optimizer_model.zero_grad()
                    first_loss.backward(retain_graph=True)
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_model.step()

        # SECOND STAGE
                if removers is not None:
                    model.eval()
                    for _, remover in removers.items():
                        remover.train()

                    t_remover_input = torch.cat((latents['pitch']['z'], latents['velocity']['z'], latents['duration']['z']), dim=1).detach()
                    p_remover_input = torch.cat((latents['timbre']['z'], latents['velocity']['z'], latents['duration']['z']), dim=1).detach()
                    v_remover_input = torch.cat((latents['timbre']['z'], latents['pitch']['z'], latents['duration']['z']), dim=1).detach()
                    l_remover_input = torch.cat((latents['timbre']['z'], latents['pitch']['z'], latents['velocity']['z']), dim=1).detach()

                    t_remover_loss = remover_loss(removers['timbre'], t_remover_input, target_classes['timbre'])
                    p_remover_loss = remover_loss(removers['pitch'], p_remover_input, target_classes['pitch'])
                    v_remover_loss = remover_loss(removers['velocity'], v_remover_input, target_classes['velocity'])
                    l_remover_loss = remover_loss(removers['duration'], l_remover_input, target_classes['duration'])
                    second_loss = (t_remover_loss + p_remover_loss + v_remover_loss + l_remover_loss) * REM_LAMBDA

                    total_removers_loss += t_remover_loss.item() + p_remover_loss.item() + v_remover_loss.item() + l_remover_loss.item()

                    if phase == 'train':
                        optimizer_removers.zero_grad()
                        second_loss.backward()
                        optimizer_removers.step()
                    model.train()
                        
            if phase == 'train':
                scheduler.step(total_loss)
                print("[LR] Last Learning Rate",scheduler.get_last_lr())
            
            if phase == 'val':
                if total_loss < best_loss:
                    print('[LOSS] PREVIOUS best loss:', best_loss)
                    best_loss = total_loss
                    print("[MODEL] Saving model...")
                    torch.save(model.state_dict(), MODELPATH)
                    print('[MODEL] Model saved!!')
                    print(f'[LOSS] ACTUAL best loss: {best_loss}')
                early_stopping(total_loss / len(dataloaders[phase]))

            if early_stopping.early_stop:
                print(f"[EARLY_STOPPING] Training stopped on epoch: {epoch}")
                return

            if LOGS:
                wandb.log({f"{phase}/global_loss": total_loss / len(dataloaders[phase]),
                            f"{phase}/reconstruction_loss": recon_loss / len(dataloaders[phase]),
                            f"{phase}/timbre_kld": -1 * total_timbre_kld / len(dataloaders[phase]),
                            f"{phase}/pitch_kld": -1 * total_pitch_kld / len(dataloaders[phase]),
                            f"{phase}/velocity_kld": -1 * total_velocity_kld / len(dataloaders[phase]),
                            f"{phase}/duration_kld": -1 * total_duration_kld / len(dataloaders[phase]),
                            f"{phase}/timbre_loss": total_timbre_loss / len(dataloaders[phase]),
                            f"{phase}/pitch_loss": total_pitch_loss / len(dataloaders[phase]),
                            f"{phase}/velocity_loss": total_velocity_loss / len(dataloaders[phase]),
                            f"{phase}/duration_loss": total_duration_loss / len(dataloaders[phase]),
                            f"{phase}/remover_KL": total_removers_kl_loss / len(dataloaders[phase]),
                            f"{phase}/remover_loss": total_removers_loss / len(dataloaders[phase])
                            })

                if phase == 'val':
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    axs[0].imshow(spectrogram[0].squeeze(0).cpu().detach().numpy(), cmap='inferno', origin='lower')
                    axs[0].set_title("Original Spectrogram")
                    axs[1].imshow(x_predict[0].squeeze(0).cpu().detach().numpy(), cmap='inferno', origin='lower')
                    axs[1].set_title("Reconstructed Spectrogram")
                    wandb.log({"MDVAE_spectrograms": wandb.Image(plt)})
                    plt.close(fig)
                else:
                    wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

    
    
def test_multidescriptor(model, dataloaders, removers=None):
    model.to(device)
    model.load_state_dict(torch.load(MODELPATH))
    model.eval()
    phase = 'test'
    
    print(f'Phase: {phase}')
    
    total_loss = 0
    total_lowerbound = 0
    total_logpx_z = 0
    total_timbre_kld_y = 0
    total_timbre_kld_z = 0
    total_pitch_kld_y = 0
    total_pitch_kld_z = 0
    total_velocity_kld_y = 0
    total_velocity_kld_z = 0
    total_duration_kld_y = 0
    total_duration_kld_z = 0
    total_timbre_h_y = 0
    total_pitch_h_y = 0
    total_velocity_h_y = 0
    total_duration_h_y = 0
    total_label_loss = 0
    total_pitch_classify_loss = 0
    total_velocity_classify_loss = 0
    total_duration_classify_loss = 0
    total_simple_mse_loss = 0

    all_timbre_true = []
    all_timbre_pred = []
    all_pitch_true = []
    all_pitch_pred = []
    all_velocity_true = []
    all_velocity_pred = []
    all_duration_true = []
    all_duration_pred = []
    
    for S, _, T, P, D, L in tqdm(dataloaders[phase]):
        spectrogram = S.to(device)

        target_classes = {'timbre': T.to(device),
                          'pitch': P.to(device),
                          'velocity': D.to(device),
                          'duration': L.to(device)
                          }

        # Forward pass
        x_predict, latents = model(spectrogram)

        # Reconstruction loss (Weighted MSE + Huber loss)
        loss1 = F.mse_loss(x_predict, spectrogram)
        loss2 = F.huber_loss(x_predict, spectrogram)
        logpx_z = -1 * (loss1 + loss2)

        # Compute the kl_latent for each descriptor
        log_q_y_logit = {}
        q_y = {}
        neg_kld_z = {}
        for name in target_classes.keys():
            log_q_y_logit[name], q_y[name], _ = model.infer_descriptor_class(latents[name][-1], descriptor=name)
            neg_kld_z[name] = -1 * kl_latent(latents[name][0], latents[name][1], q_y[name], model.mu_lookup[name], model.logvar_lookup[name])
            
        # Compute the kl_class for each descriptor
        kld_y = {}
        h_y = {}
        neg_kld_y = {}
        for name in target_classes.keys():
            kld_y[name], h_y[name] = kl_class(log_q_y_logit[name], q_y[name], model.latent_classes[name])
            neg_kld_y[name] = -1 * kld_y[name]
            
        logpx_z = torch.mean(logpx_z, dim=0)
        
        timbre_kld_z = torch.mean(neg_kld_z['timbre'], dim=0)
        pitch_kld_z = torch.mean(neg_kld_z['pitch'], dim=0)
        velocity_kld_z = torch.mean(neg_kld_z['velocity'], dim=0)
        duration_kld_z = torch.mean(neg_kld_z['duration'], dim=0)
        
        timbre_kld_y = torch.mean(neg_kld_y['timbre'], dim=0)
        pitch_kld_y = torch.mean(neg_kld_y['pitch'], dim=0)
        velocity_kld_y = torch.mean(neg_kld_y['velocity'], dim=0)
        duration_kld_y = torch.mean(neg_kld_y['duration'], dim=0)
        
        # Compute the lower bound
        lower_bound = (logpx_z + timbre_kld_y + pitch_kld_y + velocity_kld_y + duration_kld_y + timbre_kld_z + pitch_kld_z + velocity_kld_z + duration_kld_z)
        
        # Classification loss forward pass
        class_tot_loss, class_losses = classifier_loss(target_classes, log_q_y_logit, weights={'timbre': 4, 'pitch': 2, 'velocity': 2, 'duration': 2})
        class_tot_loss = torch.mean(class_tot_loss, dim=0)
        for name in class_losses.keys():
            class_losses[name] = torch.mean(class_losses[name], dim=0)
            
        for name in h_y.keys():
            h_y[name] = torch.mean(h_y[name], dim=0)
            
        # Compute the overall loss
        overall_loss = -1*lower_bound + class_tot_loss

        simple_mse_loss = F.mse_loss(x_predict, spectrogram)
        total_simple_mse_loss += simple_mse_loss.item()
        
        total_loss += overall_loss.item()
        total_lowerbound += lower_bound.item()
        total_logpx_z += logpx_z.item()
        total_timbre_kld_y += timbre_kld_y.item()
        total_timbre_kld_z += timbre_kld_z.item()
        total_pitch_kld_y += pitch_kld_y.item()
        total_pitch_kld_z += pitch_kld_z.item()
        total_velocity_kld_y += velocity_kld_y.item()
        total_velocity_kld_z += velocity_kld_z.item()
        total_duration_kld_y += duration_kld_y.item()
        total_duration_kld_z += duration_kld_z.item()
        total_timbre_h_y += h_y['timbre'].item()
        total_pitch_h_y += h_y['pitch'].item()
        total_velocity_h_y += h_y['velocity'].item()
        total_duration_h_y += h_y['duration'].item()
        total_label_loss += class_losses['timbre'].item()
        total_pitch_classify_loss += class_losses['pitch'].item()
        total_velocity_classify_loss += class_losses['velocity'].item()
        total_duration_classify_loss += class_losses['duration'].item()

        all_timbre_true.extend(target_classes['timbre'].cpu().numpy())
        all_timbre_pred.extend(log_q_y_logit['timbre'].argmax(dim=1).cpu().numpy())
        all_pitch_true.extend(target_classes['pitch'].cpu().numpy())
        all_pitch_pred.extend(log_q_y_logit['pitch'].argmax(dim=1).cpu().numpy())
        all_velocity_true.extend(target_classes['velocity'].cpu().numpy())
        all_velocity_pred.extend(log_q_y_logit['velocity'].argmax(dim=1).cpu().numpy())
        all_duration_true.extend(target_classes['duration'].cpu().numpy())
        all_duration_pred.extend(log_q_y_logit['duration'].argmax(dim=1).cpu().numpy())

        timbre_accuracy = accuracy_score(all_timbre_true, all_timbre_pred)
        timbre_f1 = f1_score(all_timbre_true, all_timbre_pred, average='weighted')
        timbre_confusion_matrix = confusion_matrix(all_timbre_true, all_timbre_pred)

        pitch_accuracy = accuracy_score(all_pitch_true, all_pitch_pred)
        pitch_f1 = f1_score(all_pitch_true, all_pitch_pred, average='weighted')
        pitch_confusion_matrix = confusion_matrix(all_pitch_true, all_pitch_pred)

        velocity_accuracy = accuracy_score(all_velocity_true, all_velocity_pred)
        velocity_f1 = f1_score(all_velocity_true, all_velocity_pred, average='weighted')
        velocity_confusion_matrix = confusion_matrix(all_velocity_true, all_velocity_pred)

        duration_accuracy = accuracy_score(all_duration_true, all_duration_pred)
        duration_f1 = f1_score(all_duration_true, all_duration_pred, average='weighted')
        duration_confusion_matrix = confusion_matrix(all_duration_true, all_duration_pred)


    if LOGS:
        wandb.log({"test/global_loss": total_loss / len(dataloaders[phase]),
                    "test/lower_bound": total_lowerbound / len(dataloaders[phase]), 
                    "test/reconstruction_loss": -1 * total_logpx_z / len(dataloaders[phase]),
                    "test/timbre_kl_class_y": -1 * total_timbre_kld_y / len(dataloaders[phase]),
                    "test/timbre_kl_latent_z": -1 * total_timbre_kld_z / len(dataloaders[phase]),
                    "test/timbre_entropy": total_timbre_h_y / len(dataloaders[phase]),
                    "test/pitch_kl_class_y": -1 * total_pitch_kld_y / len(dataloaders[phase]),
                    "test/pitch_kl_latent_z": -1 * total_pitch_kld_z / len(dataloaders[phase]),
                    "test/pitch_entropy": total_pitch_h_y / len(dataloaders[phase]),
                    "test/velocity_kl_class_y": -1 * total_velocity_kld_y / len(dataloaders[phase]),
                    "test/velocity_kl_latent_z": -1 * total_velocity_kld_z / len(dataloaders[phase]),
                    "test/velocity_entropy": total_velocity_h_y / len(dataloaders[phase]),
                    "test/duration_kl_class_y": -1 * total_duration_kld_y / len(dataloaders[phase]),
                    "test/duration_kl_latent_z": -1 * total_duration_kld_z / len(dataloaders[phase]),
                    "test/duration_entropy": total_duration_h_y / len(dataloaders[phase]),
                    "test/classifier_loss": total_label_loss / len(dataloaders[phase]),
                    "test/timbre_loss": total_label_loss / len(dataloaders[phase]), 
                    "test/pitch_loss": total_pitch_classify_loss / len(dataloaders[phase]), 
                    "test/velocity_loss": total_velocity_classify_loss / len(dataloaders[phase]),
                    "test/duration_loss": total_duration_classify_loss / len(dataloaders[phase]),
                    "test/simple_mse_loss": total_simple_mse_loss/ len(dataloaders[phase]),
                    })
            

    if METRICS:

        timbre_accuracy = accuracy_score(all_timbre_true, all_timbre_pred)
        timbre_f1 = f1_score(all_timbre_true, all_timbre_pred, average='weighted')
        timbre_confusion_matrix = confusion_matrix(all_timbre_true, all_timbre_pred)

        pitch_accuracy = accuracy_score(all_pitch_true, all_pitch_pred)
        pitch_f1 = f1_score(all_pitch_true, all_pitch_pred, average='weighted')
        pitch_confusion_matrix = confusion_matrix(all_pitch_true, all_pitch_pred)

        velocity_accuracy = accuracy_score(all_velocity_true, all_velocity_pred)
        velocity_f1 = f1_score(all_velocity_true, all_velocity_pred, average='weighted')
        velocity_confusion_matrix = confusion_matrix(all_velocity_true, all_velocity_pred)

        duration_accuracy = accuracy_score(all_duration_true, all_duration_pred)
        duration_f1 = f1_score(all_duration_true, all_duration_pred, average='weighted')
        duration_confusion_matrix = confusion_matrix(all_duration_true, all_duration_pred)

        # Save confusion matrices as .png files
        '''plot_confusion_matrix(timbre_confusion_matrix, title='MULTIDESCRIPTOR_Timbre_Confusion_Matrix', class_labels=timbre_class_names, confusion_path=CONFUSION_PATH)
        plot_confusion_matrix(pitch_confusion_matrix, title='MULTIDESCRIPTOR_Pitch_Confusion_Matrix', class_labels=pitch_class_names, confusion_path=CONFUSION_PATH)
        plot_confusion_matrix(velocity_confusion_matrix, title='MULTIDESCRIPTOR_Velocity_Confusion_Matrix', class_labels=velocity_class_names, confusion_path=CONFUSION_PATH)
        plot_confusion_matrix(duration_confusion_matrix, title='MULTIDESCRIPTOR_Duration_Confusion_Matrix', class_labels=duration_class_names, confusion_path=CONFUSION_PATH)'''

        wandb.log({"test/timbre_accuracy": timbre_accuracy, "test/timbre_f1": timbre_f1, 
                    "test/pitch_accuracy": pitch_accuracy, "test/pitch_f1": pitch_f1,
                    "test/velocity_accuracy": velocity_accuracy, "test/velocity_f1": velocity_f1,
                    "test/duration_accuracy": duration_accuracy, "test/duration_f1": duration_f1
                    })
        
        
def main():
    '''Main function to train and test the model'''
    timbre_latent_dim = get_subconfig('latent_dims').get('timbre')
    pitch_latent_dim = get_subconfig('latent_dims').get('pitch')
    velocity_latent_dim = get_subconfig('latent_dims').get('velocity')
    duration_latent_dim = get_subconfig('latent_dims').get('duration')
    bs = get_subconfig('train').get('batch_size')
    epochs = get_subconfig('train').get('num_epochs')
    lr = float(get_subconfig('train').get('learning_rate'))
    scheduler_patience = get_subconfig('train').get('scheduler_patience')
    early_stopping_patience = get_subconfig('train').get('early_stopping_patience')

    dataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/metadata.csv"))
    tinysol_dataset = TinySol_Dataset(dataframe)
    train_dataset, val_dataset, test_dataset = random_split(tinysol_dataset, [0.8, 0.1, 0.1])
 
    num_classes = {
        'timbre': tinysol_dataset.num_timbres,
        'pitch': tinysol_dataset.num_pitches,
        'velocity': tinysol_dataset.num_velocities,
        'duration': tinysol_dataset.num_durations
        }
    
    model = MultiDescriptorGMVAE(num_classes=num_classes).to(device)

    classifiers = nn.ModuleDict({
        'timbre': Classifier(timbre_latent_dim, tinysol_dataset.num_timbres),
        'pitch': Classifier(pitch_latent_dim, tinysol_dataset.num_pitches),
        'velocity': Classifier(velocity_latent_dim, tinysol_dataset.num_velocities),
        'duration': Classifier(duration_latent_dim, tinysol_dataset.num_durations)
    }).to(device)
    
    removers = nn.ModuleDict({
        'timbre': Remover(pitch_latent_dim+velocity_latent_dim+duration_latent_dim, tinysol_dataset.num_timbres),
        'pitch': Remover(timbre_latent_dim+velocity_latent_dim+duration_latent_dim, tinysol_dataset.num_pitches),
        'velocity': Remover(timbre_latent_dim+pitch_latent_dim+duration_latent_dim, tinysol_dataset.num_velocities),
        'duration': Remover(timbre_latent_dim+pitch_latent_dim+velocity_latent_dim, tinysol_dataset.num_durations)
    }).to(device)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=bs, drop_last=True, shuffle=True, num_workers=8),
        'val': DataLoader(val_dataset, batch_size=bs, drop_last=False, shuffle=False, num_workers=8),
        'test': DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=False, num_workers=8)
    }

    
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(CONFUSION_PATH):
        os.makedirs(CONFUSION_PATH)
    
    if LOGS:
        wandb.login()
        wandb.init(project="GMVAE", name="Train_GMVAE", reinit=True)
  
  
    # Train the model
    train_multidescriptor(model, 
                          dataloaders, 
                          classifiers=classifiers, 
                          removers=removers, 
                          bs=bs, 
                          num_epochs=epochs, 
                          lr=lr, 
                          scheduler_patience=scheduler_patience, 
                          early_stopping_patience=early_stopping_patience)
    
    # Test the model
    #test_multidescriptor(model, dataloaders, removers=removers)


    if LOGS:
        wandb.finish()
    
    
if __name__ == '__main__':
    main()