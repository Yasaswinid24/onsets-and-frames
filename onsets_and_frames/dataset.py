import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None,
                 seed=42, device=DEFAULT_DEVICE, feature_type='mel'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.feature_type = feature_type
        self.random = np.random.RandomState(seed)
        self.data = []

        print(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")

        for group in self.groups:
            for input_files in tqdm(self.files(group), desc=f'Loading group {group}'):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        features = data['features']
        label = data['label']
        velocity = data['velocity']

        if self.sequence_length is not None:
            n_frames = features.shape[-1]
            n_steps = self.sequence_length // HOP_LENGTH

            max_start = max(0, n_frames - n_steps)
            step_begin = self.random.randint(max_start + 1) if max_start > 0 else 0
            step_end = step_begin + n_steps

            result['features'] = features[:, step_begin:step_end].to(self.device)
            result['label'] = label[step_begin:step_end, :].to(self.device)
            result['velocity'] = velocity[step_begin:step_end, :].to(self.device)
        else:
            result['features'] = features.to(self.device)
            result['label'] = label.to(self.device)
            result['velocity'] = velocity.to(self.device)

        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        # -------------------------------
        # Load precomputed features
        # -------------------------------
        stem = audio_path.replace('.flac', '').replace('.wav', '')
        feature_path = f"{stem}.{self.feature_type}.pt"

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Missing feature file: {feature_path}\n"
                f"Run preprocess.py first."
            )

        features = torch.load(feature_path)  # (n_mels, T)

        # -------------------------------
        # Load or create labels
        # -------------------------------
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')

        if os.path.exists(saved_data_path):
            saved = torch.load(saved_data_path)
            label = saved['label']
            velocity = saved['velocity']
        else:
            audio, sr = soundfile.read(audio_path, dtype='int16')
            assert sr == SAMPLE_RATE

            audio_length = len(audio)
            n_keys = MAX_MIDI - MIN_MIDI + 1
            n_steps = (audio_length - 1) // HOP_LENGTH + 1

            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

            midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

            for onset, offset, note, vel in midi:
                left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                onset_right = min(n_steps, left + HOPS_IN_ONSET)

                frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                frame_right = min(n_steps, frame_right)

                offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                f = int(note) - MIN_MIDI
                label[left:onset_right, f] = 3
                label[onset_right:frame_right, f] = 2
                label[frame_right:offset_right, f] = 1
                velocity[left:frame_right, f] = vel

            torch.save(
                dict(path=audio_path, label=label, velocity=velocity),
                saved_data_path
            )

        return dict(
            path=audio_path,
            features=features,
            label=label,
            velocity=velocity
        )


class MAESTRO(PianoRollAudioDataset):
    def __init__(self, path='data/MAESTRO', groups=None,
                 sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE, feature_type='mel'):
        super().__init__(path,
                         groups if groups is not None else ['train'],
                         sequence_length, seed, device, feature_type)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([
                (os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                 os.path.join(self.path, row['midi_filename']))
                for row in metadata if row['split'] == group
            ])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi)
                     for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')

            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi,
                           fmt='%.6f', delimiter='\t',
                           header='onset,offset,note,velocity')

            result.append((audio_path, tsv_filename))

        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None,
                 sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE, feature_type='mel'):
        super().__init__(path,
                         groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'],
                         sequence_length, seed, device, feature_type)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb',
                'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', f'*_{group}.flac'))
        tsvs = [
            os.path.join(self.path, 'tsv', 'matched',
                         os.path.basename(flac).replace('.flac', '.tsv'))
            for flac in flacs
        ]

        assert all(os.path.isfile(flac) for flac in flacs)
        assert all(os.path.isfile(tsv) for tsv in tsvs)

        return sorted(zip(flacs, tsvs))