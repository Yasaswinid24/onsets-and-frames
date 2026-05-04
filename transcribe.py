import argparse
import os
import sys

import numpy as np
import torch
from mir_eval.util import midi_to_hz

from onsets_and_frames import *


def transcribe(model, flac_path, device):
    """
    Load precomputed features and run transcription.
    """
    feature_path = flac_path.replace('.flac', '.mel.pt').replace('.wav', '.mel.pt')

    if not os.path.exists(feature_path):
        raise RuntimeError(
            f"Missing feature file: {feature_path}\n"
            f"Run preprocess.py first."
        )

    # Load features: shape (n_mels, T)
    features = torch.load(feature_path).to(device)

    # Convert to model input: (1, T, n_mels)
    features = features.unsqueeze(0).transpose(-1, -2)

    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(features)

    predictions = {
        'onset': onset_pred.squeeze(0),
        'offset': offset_pred.squeeze(0),
        'frame': frame_pred.squeeze(0),
        'velocity': velocity_pred.squeeze(0)
    }

    return predictions


def transcribe_file(model_file, flac_paths, save_path,
                    onset_threshold, frame_threshold, device):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    for flac_path in flac_paths:
        print(f'Processing {flac_path}...', file=sys.stderr)

        predictions = transcribe(model, flac_path, device)

        p_est, i_est, v_est = extract_notes(
            predictions['onset'],
            predictions['frame'],
            predictions['velocity'],
            onset_threshold,
            frame_threshold
        )

        scaling = HOP_LENGTH / SAMPLE_RATE
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)

        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])

        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('flac_paths', type=str, nargs='+')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))