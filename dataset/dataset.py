import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TinySol_Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.num_timbres = len(self.dataframe.iloc[:, 1].unique())
        self.num_pitches = len(self.dataframe.iloc[:, 2].unique())
        self.num_velocities = len(self.dataframe.iloc[:, 3].unique())
        self.num_durations = max(self.dataframe.iloc[:, 4]) + 1

        self.min_pitch = self.dataframe.iloc[:, 2].min()
        
        self.timbre_labels = sorted(self.dataframe.iloc[:, 1].unique())
        self.pitch_labels = sorted(self.dataframe.iloc[:, 2].unique())
        self.velocity_labels = sorted(self.dataframe.iloc[:, 3].unique())
        self.duration_labels = sorted(self.dataframe.iloc[:, 4].unique())
        

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx, 0]

        timbre = torch.tensor(self.dataframe.iloc[idx, 1])
        pitch = torch.tensor(self.dataframe.iloc[idx, 2] - self.min_pitch)
        velocity = torch.tensor(self.dataframe.iloc[idx, 3])
        duration = torch.tensor(self.dataframe.iloc[idx, 4])
        spectrogram = (torch.from_numpy(np.load(path)).unsqueeze(0).float() + 80)/80
        
        return spectrogram, timbre, pitch, velocity, duration



if __name__ == '__main__':
    SAMPLE_IDX = 313
    CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.csv")
    dataframe = pd.read_csv(CSV)
    dataset = TinySol_Dataset(dataframe)
    print(dataframe.head())
    print(dataframe.keys())
    print(f'Length of dataset: {len(dataset)}')
    sample = dataset[SAMPLE_IDX]
    print('Example sample:')
    print(f'Example Spectrogram shape: {sample[0].shape}')
    print(f'Example Label: {sample[1]}')
    print(f'Example Pitch: {sample[2]}')
    print(f'Example Velocity: {sample[3]}')
    print(f'Example Duration: {sample[4]}')
    print(20*'-')
    print('Dataset info:')
    
    print(f'Number of timbre labels: {dataset.num_timbres}')
    print(f'Number of pitch labels: {dataset.num_pitches}')
    print(f'Number of velocity labels: {dataset.num_velocities}')
    print(f'Number of duration labels: {dataset.num_durations}')
    
    print(f'Minimum pitch: {dataset.min_pitch}')
    
    print(f'Timbre labels: {dataset.timbre_labels}')
    print(f'Pitch labels: {dataset.pitch_labels}')
    print(f'Velocity labels: {dataset.velocity_labels}')
    print(f'Duration labels: {dataset.duration_labels}')

    print(min(sample[0]))
    print(max(sample[0]))

