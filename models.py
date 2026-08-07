import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SliceWiseVisionEncoder(nn.Module):
    """
    2D CNN/ConvNeXt feature extractor applied across slice dimension (2.5D aggregation).
    """
    def __init__(self, embed_dim=config.EMBED_DIM):
        super(SliceWiseVisionEncoder, self).__init__()
        # Convolutional feature extractor per slice
        self.conv = nn.Sequential(
            nn.Conv2d(config.NUM_CHANNELS, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x shape: (B, D, C, H, W)
        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W)
        feats = self.conv(x).view(B * D, -1)
        feats = self.proj(feats) # (B * D, embed_dim)
        feats = feats.view(B, D, -1) # (B, D, embed_dim)
        return feats

class VolumetricSliceAggregator(nn.Module):
    """
    Sequence Transformer / Attention Pooling across 3D slice depth.
    """
    def __init__(self, embed_dim=config.EMBED_DIM):
        super(VolumetricSliceAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_weights = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (B, D, embed_dim)
        tokens = self.transformer(x) # (B, D, embed_dim)
        weights = F.softmax(self.attn_weights(tokens), dim=1) # (B, D, 1)
        pooled = torch.sum(tokens * weights, dim=1) # (B, embed_dim)
        return pooled

class MultimodalKneeClassifier(nn.Module):
    """
    Complete Multimodal Network: Fuses 3D Volume features with text report representation for 12 target classes.
    """
    def __init__(self, num_classes=config.NUM_CLASSES, embed_dim=config.EMBED_DIM):
        super(MultimodalKneeClassifier, self).__init__()
        self.slice_encoder = SliceWiseVisionEncoder(embed_dim=embed_dim)
        self.vol_aggregator = VolumetricSliceAggregator(embed_dim=embed_dim)
        
        # Classification Head for 12 Target Diseases/Abnormalities
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def forward(self, volume, report_text=None):
        # volume: (B, D, C, H, W)
        slice_feats = self.slice_encoder(volume)
        vol_feats = self.vol_aggregator(slice_feats)
        logits = self.head(vol_feats)
        return logits
