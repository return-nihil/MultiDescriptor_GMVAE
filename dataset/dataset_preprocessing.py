import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from utils import get_subconfig
config = get_subconfig('dataset_preprocessing')


def remove_silence(y):
    '''Remove silence from the beginning and end of the audio signal'''
    non_silence_indexes = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=1024)
    return np.concatenate([y[start:end] for start, end in non_silence_indexes])


def adjust_audio_length(sample, sr, apply_fade=True):
    '''Fix the length of the audio signal to a target length'''
    max_dur_sec = config.get('max_dur')
    target_length = int(sr * max_dur_sec)
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
    return librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_fft=config.get('n_fft'),
            hop_length=config.get('hop_length'),
            n_mels=config.get('n_mels')
        ),
        ref=1,
        top_db=80
    )


def build_duration_dict(max_duration_ms, quant_ms):
    '''Build a dictionary to map duration ranges to bin indices'''
    return {
        f'{i * quant_ms}-{min((i + 1) * quant_ms, max_duration_ms)}ms': i
        for i in range((max_duration_ms // quant_ms) + 1)
    }


def quantize_duration(duration_ms, quant_ms, max_duration_ms):
    '''Quantize the duration to the nearest bin'''
    return int(min(duration_ms, max_duration_ms) // quant_ms)


def parse_filename(filename):
    '''Parse the filename to extract instrument, pitch, and velocity'''
    parts = filename.split('.')[0].split('-')
    return parts[0], parts[2], parts[3]


def process():
    '''Main function to process the dataset'''
    sr = config.get('sr')
    dur_quant = config.get('dur_quant')  
    max_dur_sec = config.get('max_dur')  
    max_dur_ms = int(max_dur_sec * 1000)

    orig_folder = config.get('orig_dataset_folder')
    target_folder = config.get('target_dataset_folder')
    os.makedirs(target_folder, exist_ok=True)

    duration_dict = build_duration_dict(max_dur_ms, dur_quant)
    metadata = []
    instrument_set = set()
    velocity_set = set()

    for folder, _, files in os.walk(orig_folder):
        for file in tqdm(files, desc=f'Processing files: {folder}'):
            if not file.endswith('.wav'):
                continue

            file_path = os.path.join(folder, file)
            instrument, pitch_str, velocity = parse_filename(file)

            if velocity == 'p':  # fix bad label
                velocity = 'pp'

            instrument_set.add(instrument)
            velocity_set.add(velocity)

            y, _ = librosa.load(file_path, sr=sr)
            y = remove_silence(y)
            dur_sec = librosa.get_duration(y=y, sr=sr)
            y = adjust_audio_length(y, sr=sr)
            spec = extract_spectrogram(y, sr=sr)
            assert spec.shape == (256, 256), f'Error with file: {file}'

            output_filename = file.replace('.wav', f'-{int(dur_sec * 1000)}.npy')
            output_path = os.path.join(target_folder, output_filename)
            np.save(output_path, spec)

            pitch = librosa.note_to_midi(pitch_str)
            duration_ms = int(dur_sec * 1000)
            duration_class = quantize_duration(duration_ms, dur_quant, max_dur_ms)

            metadata.append({
                'Path': os.path.abspath(output_path),
                'Instrument': instrument,
                'Pitch': pitch,
                'Velocity': velocity,
                'Duration': duration_class
            })

    instrument_dict = {k: i for i, k in enumerate(sorted(instrument_set))}
    velocity_dict = {k: i for i, k in enumerate(sorted(velocity_set))}

    df_rows = []
    for entry in metadata:
        df_rows.append({
            'Path': entry['Path'],
            'Timbre': instrument_dict[entry['Instrument']],
            'Pitch': entry['Pitch'],
            'Velocity': velocity_dict[entry['Velocity']],
            'Duration': entry['Duration']
        })

    df = pd.DataFrame(df_rows)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df.to_csv(os.path.join(script_dir, 'metadata.csv'), index=False)

    with open(os.path.join(script_dir, 'class_mappings.json'), 'w') as f:
        json.dump({
            'instrument_dict': instrument_dict,
            'velocity_dict': velocity_dict,
            'duration_dict': duration_dict
        }, f, indent=2)

    print(f"Metadata and class mappings saved to {script_dir}")



if __name__ == '__main__':
    process()