import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import config

def seed_everything(seed=config.SEED):
    """Seed all random number generators for exact reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_macro_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged AUC-ROC score across all 12 abnormality target columns.
    
    Args:
        y_true: np.ndarray of shape (N, 12) - Ground truth binary targets
        y_pred: np.ndarray of shape (N, 12) - Predicted probability scores
        
    Returns:
        dict: Overall macro AUC score and individual column AUC scores
    """
    column_aucs = {}
    auc_list = []
    
    for i, col in enumerate(config.TARGET_COLS):
        try:
            # Calculate ROC AUC per column if both positive and negative samples exist
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                column_aucs[col] = score
                auc_list.append(score)
            else:
                column_aucs[col] = 0.5
        except Exception:
            column_aucs[col] = 0.5

    macro_auc = np.mean(auc_list) if len(auc_list) > 0 else 0.5
    return macro_auc, column_aucs

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Effectively handles severe positive/negative class imbalance in medical image targets.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # Sigmoid activation
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic BCE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_pos > 0:
            los_pos = los_pos * ((1 - xs_pos) ** self.gamma_pos)
        if self.gamma_neg > 0:
            los_neg = los_neg * ((xs_pos) ** self.gamma_neg)

        loss = - (los_pos + los_neg)
        return loss.mean()
