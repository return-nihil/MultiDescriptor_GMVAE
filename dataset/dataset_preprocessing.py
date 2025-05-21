import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
from tqdm import tqdm
from utils import get_subconfig
config = get_subconfig('dataset_preprocessing')


def remove_silence(y):
    '''Remove silence from the beginning and end of the audio signal'''
    non_silence_indexes = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=1024)
    y_trimmed = np.concatenate([y[start:end] for start, end in non_silence_indexes])
    return y_trimmed


def adjust_audio_length(sample, sr, apply_fade=True):
    '''Fix the length of the audio signal to a target length'''
    max_duration = config.get('max_duration')

    target_length = int(sr * max_duration)
    length = sample.shape[0]

    if length > target_length:
        sample = sample[:target_length]
        if apply_fade:
            envelope_duration = int(0.5 * sr)
            fade = np.exp(-np.linspace(0, 0.5, envelope_duration))
            sample[-envelope_duration:] *= fade
    elif length < target_length:
        padding = np.zeros(target_length - length, dtype=sample.dtype)
        sample = np.concatenate((sample, padding))

    return sample
    

def extract_spectrogram(sample, sr):
    '''Extract the mel spectrogram from the audio signal'''
    n_fft = config.get('n_fft')
    hop_length = config.get('hop_length')
    n_mels = config.get('n_mels')

    spectrogram = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    return librosa.power_to_db(spectrogram, ref=1, top_db=80)


def process():
    '''Process the audio files in the dataset'''
    sr = config.get('sr')
    orig_dataset_folder = config.get('orig_dataset_folder')
    target_dataset_folder = config.get('target_dataset_folder')
    os.mkdir(target_dataset_folder) if not os.path.exists(target_dataset_folder) else None
    
    for folder, _, files in os.walk(orig_dataset_folder):
        for file in tqdm(files, desc=f'Processing files: {folder}'):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                y, _ = librosa.load(file_path, sr=sr)
                y = remove_silence(y)
                y = adjust_audio_length(y, sr=sr)
                dur_sec = librosa.get_duration(y=y, sr=sr) 
                spec = extract_spectrogram(y, sr=sr)
                
                assert spec.shape == (256, 256), f'Error with file: {file}'
                file = file.split('.')[0] + '-' + str(int(dur_sec*1000)) + '.wav'
                
                numpy_file = file.replace('.wav', '.npy')
                dest = os.path.join(target_dataset_folder, numpy_file)
                np.save(dest, spec)

           
if __name__ == "__main__":
    process()

