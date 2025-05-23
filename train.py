import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
thisdir = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb

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

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import get_subconfig, xavier_init, kaiming_init
from plotters import log_spectrograms, plot_confusion_matrix, plot_latent_space


data_config = get_subconfig('data')
train_config = get_subconfig('train')
device = get_subconfig('device')
OUTPUT_FOLDER = get_subconfig('train').get('output_folder_name')
if not os.path.exists(os.path.join(thisdir, OUTPUT_FOLDER)):
        os.makedirs(os.path.join(thisdir, OUTPUT_FOLDER))
LOGS = get_subconfig('train').get('logs')
if LOGS:
    wandb.login()
    wandb.init(project="GMVAE", name="Train_GMVAE", reinit=True)

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

      
def train(model, 
        dataloaders, 
        classifiers, 
        removers, 
        num_epochs, 
        lr, 
        recon_loss_fn,
        kl_emb_loss_fn,
        spread_loss_fn,
        classifier_loss_fn,
        remover_loss_fn,
        remover_kl_loss_fn,
        beta_warmup,
        pretrain_no_removers,
        scheduler_patience, 
        early_stopping_patience
        ):
    model.apply(xavier_init)
    model.to(device)
    
    for _, remover in removers.items():
        remover.apply(kaiming_init)
    optimizer_removers = Adam(
        [param for remover in removers.values() for param in remover.parameters()], 
        weight_decay=1e-5,
        lr=lr)
        
    optimizer_model = Adam(
        list(model.parameters()) + 
        [p for clf in classifiers.values() for p in clf.parameters()],
        weight_decay=1e-5,
        lr=lr
    )
    
    scheduler = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-6) 
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.00001)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('-' * 50)
        print(f'[EPOCH] {epoch+1}/{num_epochs}')

        for phase in ['train', 'val']:
            print(f'Phase: {phase}')

            total_loss = 0
            total_recon = 0
            total_spread_loss = 0
            total_removers_KL_loss = 0
            total_removers_loss = 0
    
            for S, T, P, V, D in tqdm(dataloaders[phase]):
                spectrogram, timbre_label, pitch_label, velocity_label, duration_label = S.to(device), T.to(device), P.to(device), V.to(device), D.to(device)
                
                descriptors = ['timbre', 'pitch', 'velocity', 'duration']
                target_classes = {
                    'timbre': timbre_label,
                    'pitch': pitch_label,
                    'velocity': velocity_label,
                    'duration': duration_label
                }

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                noisy_spectrogram = spectrogram + torch.randn_like(spectrogram) * 1e-4
                x_predict, latents = model(noisy_spectrogram)
                
                recon_loss = recon_loss_fn(x_predict, spectrogram)

                kl_losses = {}
                spread_losses = {}
                class_losses = {}
                loss_totals = {f'total_{desc}_kld': 0 for desc in descriptors}
                loss_totals.update({f'total_{desc}_loss': 0 for desc in descriptors})

                for desc in descriptors:
                    kl_losses[desc] = kl_emb_loss_fn(
                        model.mu_lookup[desc], model.logvar_lookup[desc], 
                        latents[desc]['mu'], latents[desc]['logvar'], 
                        target_classes[desc]
                    )

                    spread_losses[desc] = spread_loss_fn(latents[desc]['z'])
                    
                    class_pred = classifiers[desc](latents[desc]['z'])
                    class_losses[desc] = classifier_loss_fn(class_pred, target_classes[desc])

                beta = epoch/beta_warmup if epoch < beta_warmup else 1
                total_loss = (recon_loss + 
                            beta * sum(kl_losses[desc].mean() for desc in descriptors) +
                            sum(spread_losses[desc] for desc in descriptors) +
                            sum(class_losses[desc] for desc in descriptors))

                remover_inputs = {
                    'timbre': torch.cat([latents[d]['z'] for d in ['pitch', 'velocity', 'duration']], dim=1),
                    'pitch': torch.cat([latents[d]['z'] for d in ['timbre', 'velocity', 'duration']], dim=1),
                    'velocity': torch.cat([latents[d]['z'] for d in ['timbre', 'pitch', 'duration']], dim=1),
                    'duration': torch.cat([latents[d]['z'] for d in ['timbre', 'pitch', 'velocity']], dim=1)
                }

                # FIRST STAGE
                if epoch >= pretrain_no_removers:
                    beta_rem = (epoch-pretrain_no_removers)/(beta_warmup if epoch < beta_warmup else 1)
                    for _, remover in removers.items():
                        remover.eval()

                    remover_kl_losses = {}
                    for desc in descriptors:
                        rem_logits = removers[desc](remover_inputs[desc])
                        num_classes = model.num_classes[desc]
                        remover_kl_losses[desc] = remover_kl_loss_fn(rem_logits, num_classes)
                else:
                    beta_rem = 0
                    remover_kl_losses = {desc: torch.zeros(1).to(device) for desc in descriptors}

                first_loss = total_loss + sum(remover_kl_losses.values()) * beta_rem

                total_loss += first_loss.item()
                total_recon += recon_loss.item()
                total_spread_loss += sum(spread_losses[desc].item() for desc in descriptors)
                
                for desc in descriptors:
                    loss_totals[f'total_{desc}_kld'] += kl_losses[desc].sum().item()
                    loss_totals[f'total_{desc}_loss'] += class_losses[desc].item()
                
                total_removers_KL_loss += sum(loss.item() for loss in remover_kl_losses.values())

                if phase == 'train':
                    optimizer_model.zero_grad()
                    first_loss.backward(retain_graph=True)
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_model.step()

                # SECOND STAGE
                model.eval()
                for _, remover in removers.items():
                    remover.train()

                detached_inputs = {desc: inp.detach() for desc, inp in remover_inputs.items()}
                remover_losses = {}
                for desc in descriptors:
                    rem_logits = removers[desc](detached_inputs[desc])
                    remover_losses[desc] = remover_loss_fn(rem_logits, target_classes[desc])
                
                second_loss = sum(remover_losses.values()) 
                total_removers_loss += sum(loss.item() for loss in remover_losses.values())

                if phase == 'train':
                    optimizer_removers.zero_grad()
                    second_loss.backward()
                    optimizer_removers.step()
                model.train()
                        
            if phase == 'train':
                scheduler.step(total_loss)

            if phase == 'val':
                print(f'[LOSS RECON]: {total_recon / len(dataloaders[phase])}')
                print(f'[LOSS]: {total_loss / len(dataloaders[phase])}')
                print("[LR]", scheduler.get_last_lr())
                if total_loss < best_loss:
                    best_loss = total_loss
                    print("[MODEL] Saving...")
                    torch.save(model.state_dict(), os.path.join(thisdir, OUTPUT_FOLDER, 'MD_GMVAE.pth'))
                early_stopping(total_loss / len(dataloaders[phase]))

            if early_stopping.early_stop:
                print(f"[EARLY_STOPPING]: {epoch}")
                return

            if LOGS:
                log_dict = {
                    f"{phase}/global_loss": total_loss / len(dataloaders[phase]),
                    f"{phase}/reconstruction_loss": total_recon / len(dataloaders[phase]),
                    f"{phase}/spread_loss": total_spread_loss / len(dataloaders[phase]),
                    f"{phase}/remover_KL_loss": total_removers_KL_loss / len(dataloaders[phase]),
                    f"{phase}/remover_loss": total_removers_loss / len(dataloaders[phase]),
                }
                for desc in descriptors:
                    log_dict.update({
                        f"{phase}/{desc}_kl_loss": loss_totals[f'total_{desc}_kld'] / len(dataloaders[phase]),
                        f"{phase}/{desc}_loss": loss_totals[f'total_{desc}_loss'] / len(dataloaders[phase]),
                    })
                
                wandb.log(log_dict)
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

                if phase == 'val' and epoch % 5 == 0:
                    log_spectrograms(spectrogram, x_predict)


def test(model, 
         dataloaders, 
         classifiers, 
         recon_loss_fn, 
         kl_emb_loss_fn, 
         classifier_loss_fn):
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(thisdir, OUTPUT_FOLDER, 'MD_GMVAE.pth')))
    model.eval()
    phase = 'test'
    
    print(f'Phase: {phase}')
    
    descriptors = ['timbre', 'pitch', 'velocity', 'duration']
    
    total_loss = 0
    total_recon = 0
    total_samples = 0
    
    total_kl = {desc: 0 for desc in descriptors}
    total_classifier_loss = {desc: 0 for desc in descriptors}
    
    all_true = {desc: [] for desc in descriptors}
    all_pred = {desc: [] for desc in descriptors}
    all_latents = {desc: [] for desc in descriptors}

    for S, T, P, V, D in tqdm(dataloaders[phase]):
        spectrogram = S.to(device)
        timbre_label = T.to(device)
        pitch_label = P.to(device)
        velocity_label = V.to(device)
        duration_label = D.to(device)

        target_classes = {
            'timbre': timbre_label,
            'pitch': pitch_label,
            'velocity': velocity_label,
            'duration': duration_label
        }

        with torch.no_grad():
            x_predict, latents = model(spectrogram)
            recon_loss = recon_loss_fn(x_predict, spectrogram)

            kl_losses = {}
            class_losses = {}
            class_preds = {}

            for desc in descriptors:
                kl_losses[desc] = kl_emb_loss_fn(
                    model.mu_lookup[desc], model.logvar_lookup[desc], 
                    latents[desc]['mu'], latents[desc]['logvar'], 
                    target_classes[desc]
                )
                class_pred = classifiers[desc](latents[desc]['z'])  # logits for desc
                class_preds[desc] = class_pred
                class_losses[desc] = classifier_loss_fn(class_pred, target_classes[desc])
            
            total_loss_batch = recon_loss + sum(kl_losses[desc].mean() for desc in kl_losses) + sum(class_losses[desc] for desc in class_losses)

            total_loss += total_loss_batch.item()
            total_recon += recon_loss.item()
            total_samples += spectrogram.size(0)

            for desc in descriptors:
                total_kl[desc] += kl_losses[desc].sum().item()
                total_classifier_loss[desc] += class_losses[desc].item()
            for desc in descriptors:
                preds_np = class_preds[desc].argmax(dim=1).cpu().numpy()
                true_np = target_classes[desc].cpu().numpy()
                all_pred[desc].extend(preds_np)
                all_true[desc].extend(true_np)
                all_latents[desc].append(latents[desc]['z'].cpu().numpy())

    print(f"Avg reconstruction loss: {total_recon / total_samples:.4f}")
    print(f"Avg total loss: {total_loss / total_samples:.4f}")

    if LOGS:
        log_dict = {
            "test/global_loss": total_loss / total_samples,
            "test/reconstruction_loss": total_recon / total_samples,
        }
        for desc in descriptors:
            log_dict[f"test/{desc}_kl_loss"] = total_kl[desc] / total_samples
            log_dict[f"test/{desc}_classifier_loss"] = total_classifier_loss[desc] / total_samples
        wandb.log(log_dict)

        for desc in descriptors:
            all_latents[desc] = np.concatenate(all_latents[desc], axis=0)
            all_true[desc] = np.array(all_true[desc])

            accuracy = accuracy_score(all_true[desc], all_pred[desc])
            f1 = f1_score(all_true[desc], all_pred[desc], average='weighted')
            confusion = confusion_matrix(all_true[desc], all_pred[desc])

            plot_confusion_matrix(
                confusion,
                descriptor=desc,
                output_path=os.path.join(thisdir, OUTPUT_FOLDER)
            )

            plot_latent_space(
                all_latents[desc],
                all_true[desc],
                descriptor=desc,
                output_path=os.path.join(thisdir, OUTPUT_FOLDER),
                filename_prefix="test_latents"
            )

            print(f"{desc} accuracy: {accuracy:.4f}, f1: {f1:.4f}")
            wandb.log({
                f"test/{desc}_accuracy": accuracy,
                f"test/{desc}_f1": f1
            })



        
def main():
    '''Main function to train and test the model'''
    timbre_latent_dim =         get_subconfig('latent_dims').get('timbre')
    pitch_latent_dim =          get_subconfig('latent_dims').get('pitch')
    velocity_latent_dim =       get_subconfig('latent_dims').get('velocity')
    duration_latent_dim =       get_subconfig('latent_dims').get('duration')
    min_pitch, max_pitch =      get_subconfig('train').get('pitch_range')
    bs =                        get_subconfig('train').get('batch_size')
    epochs =                    get_subconfig('train').get('num_epochs')
    lr =                        float(get_subconfig('train').get('learning_rate'))
    beta_warmup =               get_subconfig('train').get('beta_warmup')
    pretrain_no_removers =      get_subconfig('train').get('pretrain_no_removers')
    scheduler_patience =        get_subconfig('train').get('scheduler_patience')
    early_stopping_patience =   get_subconfig('train').get('early_stopping_patience')


    dataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/metadata.csv"))
    filtered_df = dataframe[(dataframe['Pitch'] >= min_pitch) & (dataframe['Pitch'] <= max_pitch)]
    tinysol_dataset = TinySol_Dataset(filtered_df)
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

    recon_loss_fn = Reconstruction_Loss()
    kl_emb_loss_fn = KL_Emb_Loss()
    spread_loss_fn = Latent_Spread_Loss()
    classifier_loss_fn = Classifier_Loss()
    remover_loss_fn = Remover_Loss()
    remover_kl_loss_fn = Remover_KL_Uniform_Loss()
  
    train(model=model, 
            dataloaders=dataloaders, 
            classifiers=classifiers, 
            removers=removers, 
            num_epochs=epochs, 
            lr=lr, 
            recon_loss_fn=recon_loss_fn,
            kl_emb_loss_fn=kl_emb_loss_fn,
            spread_loss_fn=spread_loss_fn,
            classifier_loss_fn=classifier_loss_fn,
            remover_loss_fn=remover_loss_fn,
            remover_kl_loss_fn=remover_kl_loss_fn,
            beta_warmup=beta_warmup,
            pretrain_no_removers=pretrain_no_removers,
            scheduler_patience=scheduler_patience, 
            early_stopping_patience=early_stopping_patience
            )
    
    test(model=model, 
         dataloaders=dataloaders, 
         classifiers=classifiers,
         recon_loss_fn=recon_loss_fn, 
         kl_emb_loss_fn=kl_emb_loss_fn, 
         classifier_loss_fn=classifier_loss_fn)


    print("Extracting latents from full dataset and plotting t-SNEs...")
    full_dataloader = DataLoader(tinysol_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)

    descriptors = ['timbre', 'pitch', 'velocity', 'duration']
    all_latents = {desc: [] for desc in descriptors}
    all_labels = {desc: [] for desc in descriptors}

    model.eval()
    for S, T, P, V, D in tqdm(full_dataloader):
        S = S.to(device)
        labels = {'timbre': T, 'pitch': P, 'velocity': V, 'duration': D}

        with torch.no_grad():
            _, latents = model(S)

        for desc in descriptors:
            all_latents[desc].append(latents[desc]['z'].cpu().numpy())
            all_labels[desc].append(labels[desc].numpy())

    for desc in descriptors:
        latents_np = np.concatenate(all_latents[desc], axis=0)
        labels_np = np.concatenate(all_labels[desc], axis=0)

        plot_latent_space(
            latents=latents_np,
            true_labels=labels_np,
            descriptor=desc,
            output_path=os.path.join(thisdir, OUTPUT_FOLDER),
            filename_prefix="all_latents"
        )

    if LOGS:
        wandb.finish()
    
    
if __name__ == '__main__':
    main()