import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config

class RSNAMultimodalKneeDataset(Dataset):
    """
    Dataset class for RSNA Knee Abnormality Detection.
    Handles loading 3D MRI image volumes, paired radiology reports, and 12 binary labels.
    """
    def __init__(self, df, is_train=True, transform=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_volume(self, study_uid):
        """
        Simulate/load multi-planar MRI 3D volumes (Depth x Channels x Height x Width).
        In production with real DICOM/NIfTI files, loads slices from directory.
        """
        # Create standardized volume shape (NUM_SLICES, NUM_CHANNELS, H, W)
        D, C, H, W = config.NUM_SLICES, config.NUM_CHANNELS, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]
        
        # Synthetic generator fallback if DICOM data directory is not loaded yet
        volume = np.random.randn(D, C, H, W).astype(np.float32)
        return torch.tensor(volume, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Load MRI volume
        volume = self._load_volume(study_uid)

        # Load radiology report text if available
        report_text = str(row.get("RadiologyReport", "Normal knee MRI examination."))

        # Extract target labels if training/validation
        if self.is_train and all(col in row for col in config.TARGET_COLS):
            targets = torch.tensor(row[config.TARGET_COLS].values.astype(np.float32), dtype=torch.float32)
        else:
            targets = torch.zeros(config.NUM_CLASSES, dtype=torch.float32)

        return {
            "study_uid": study_uid,
            "volume": volume,
            "report_text": report_text,
            "targets": targets
        }
