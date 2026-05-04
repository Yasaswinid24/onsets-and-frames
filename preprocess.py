"""
preprocess.py  –  Compute and cache audio features for onsets-and-frames.

Usage
-----
# Precompute mel spectrograms for the MAPS dataset (default):
    python preprocess.py

# Specific groups:
    python preprocess.py --groups ENSTDkAm ENSTDkCl

# MAESTRO dataset:
    python preprocess.py --dataset-path data/MAESTRO --dataset MAESTRO

# Different feature type (extend compute_features() to add more):
    python preprocess.py --feature-type mel

Output
------
One  <audio_stem>.<feature_type>.pt  file written next to each audio file.
The tensor stored inside has shape  (n_mels, T)  – identical to what
dataset.py would produce via  self.mel(audio.unsqueeze(0)[:, :-1]).squeeze(0).
"""

import argparse
import os
import sys

import numpy as np
import soundfile
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make the package importable when running from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from onsets_and_frames.constants import (
    DEFAULT_DEVICE, HOP_LENGTH, MEL_FMAX, MEL_FMIN,
    N_MELS, SAMPLE_RATE, WINDOW_LENGTH,
)
from onsets_and_frames.mel import MelSpectrogram
import onsets_and_frames.dataset as dataset_module


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def build_feature_extractor(feature_type: str, device: str):
    """
    Return a callable  f(audio_1d_float_tensor) -> feature_tensor.

    Adding a new feature type is a single new branch here; nothing else in
    the script needs to change.

    Parameters
    ----------
    feature_type : str
        One of: 'mel'  (more can be added: 'pcen', 'linear', 'log', …)
    device : str
        Torch device string, e.g. 'cpu' or 'cuda'.

    Returns
    -------
    extractor : callable
        Accepts a 1-D float tensor in [-1, 1], returns a 2-D feature tensor.
    """
    if feature_type == "mel":
        mel_transform = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH,
            mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX,
        ).to(device)

        def extractor(audio: torch.Tensor) -> torch.Tensor:
            # Replicate dataset.py exactly:
            #   self.mel(result['audio'].unsqueeze(0)[:, :-1]).squeeze(0)
            # audio is 1-D float tensor in [-1, 1]
            with torch.no_grad():
                features = mel_transform(
                    audio.unsqueeze(0)[:, :-1]   # drop last sample, add batch dim
                ).squeeze(0)                      # → (n_mels, T)
            return features

        return extractor

    # ------------------------------------------------------------------
    # Placeholder for future feature types – add branches here:
    #
    # elif feature_type == "pcen":
    #     ...
    #     return extractor
    #
    # elif feature_type == "linear":
    #     ...
    #     return extractor
    # ------------------------------------------------------------------

    raise ValueError(
        f"Unknown feature type '{feature_type}'. "
        f"Supported types: mel"
    )


# ---------------------------------------------------------------------------
# Audio loading  (mirrors dataset.py exactly)
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, device: str) -> torch.Tensor:
    """
    Load an audio file and return a normalised float tensor in [-1, 1],
    matching the normalisation applied in PianoRollAudioDataset.__getitem__:

        audio.float().div_(32768.0)

    Parameters
    ----------
    audio_path : str
    device : str

    Returns
    -------
    torch.FloatTensor, shape = [num_samples]
    """
    audio, sr = soundfile.read(audio_path, dtype='int16')
    assert sr == SAMPLE_RATE, (
        f"Expected sample rate {SAMPLE_RATE}, got {sr} for {audio_path}"
    )
    audio = torch.ShortTensor(audio).to(device)
    audio = audio.float().div_(32768.0)          # normalise to [-1, 1]
    return audio


# ---------------------------------------------------------------------------
# Core preprocessing loop
# ---------------------------------------------------------------------------

def preprocess_group(
    dataset_path: str,
    dataset_name: str,
    group: str,
    feature_type: str,
    extractor,
    device: str,
):
    """
    Compute and save features for every audio file in *group*.

    Parameters
    ----------
    dataset_path  : root path passed to the dataset class
    dataset_name  : 'MAESTRO' or 'MAPS'
    group         : group name (e.g. 'train', 'ENSTDkAm')
    feature_type  : string key used to name the output file
    extractor     : callable returned by build_feature_extractor()
    device        : torch device string
    """
    # Resolve (audio_path, tsv_path) pairs via the existing dataset machinery
    dataset_class = getattr(dataset_module, dataset_name)
    ds = dataset_class.__new__(dataset_class)
    ds.path = dataset_path

    file_pairs = list(ds.files(group))      # [(audio_path, tsv_path), …]

    skipped = 0
    processed = 0

    for audio_path, _tsv_path in tqdm(file_pairs, desc=f"{dataset_name}/{group}"):
        stem, _ = os.path.splitext(audio_path)
        out_path = f"{stem}.{feature_type}.pt"

        if os.path.exists(out_path):
            skipped += 1
            continue

        audio = load_audio(audio_path, device)
        features = extractor(audio)          # 2-D tensor, e.g. (n_mels, T)

        torch.save(features, out_path)
        processed += 1

    print(
        f"  {dataset_name}/{group}: "
        f"{processed} computed, {skipped} skipped (already existed)."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute audio features and save as .pt files."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/MAPS",
        help="Root directory of the dataset (default: data/MAPS).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MAPS",
        choices=["MAESTRO", "MAPS"],
        help="Dataset class to use (default: MAPS).",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Groups to process.  Defaults to ['train'] for MAESTRO and "
            "['ENSTDkAm', 'ENSTDkCl'] for MAPS.  Pass 'train validation test' "
            "to process all MAESTRO splits."
        ),
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="mel",
        help="Feature type to compute (default: mel).  Currently supported: mel.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device (default: {DEFAULT_DEVICE}).",
    )

    args = parser.parse_args()

    # Resolve default groups per dataset
    if args.groups is None:
        defaults = {
            "MAESTRO": ["train"],
            "MAPS": ["ENSTDkAm", "ENSTDkCl"],
        }
        args.groups = defaults[args.dataset]

    print(
        f"Preprocessing  dataset={args.dataset}  "
        f"groups={args.groups}  feature={args.feature_type}  "
        f"device={args.device}"
    )

    extractor = build_feature_extractor(args.feature_type, args.device)

    for group in args.groups:
        preprocess_group(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset,
            group=group,
            feature_type=args.feature_type,
            extractor=extractor,
            device=args.device,
        )

    print("Done.")


if __name__ == "__main__":
    main()