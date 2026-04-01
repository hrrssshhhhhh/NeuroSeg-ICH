"""
ResUNet - Residual U-Net for Medical Image Segmentation
================================================
Author  : Harsh
Task    : Binary segmentation (Intracranial Hemorrhage on CT scans)
Input   : RGB CT slices  [B, 3, H, W]
Output  : Logits         [B, 1, H, W]  (apply sigmoid at inference)
"""

import torch
import torch.nn as nn


# =============================================================
# BLOCK 1 — Batch Norm + ReLU + Conv (pre-activation style)
# =============================================================

class BNReLUConv(nn.Module):
    """BatchNorm → ReLU → Conv2D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False)
        )

    def forward(self, x):
        return self.block(x)


# =============================================================
# BLOCK 2 — Residual Block (the core of ResUNet)
# =============================================================

class ResidualBlock(nn.Module):
    """
    Two BNReLUConv layers with a residual shortcut.
    Shortcut uses 1x1 conv when channel size changes.

    x ──► BNReLUConv ──► BNReLUConv ──► (+) ──► out
    │                                     ▲
    └────────── shortcut (1x1 conv) ──────┘
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = BNReLUConv(in_channels, out_channels)
        self.layer2 = BNReLUConv(out_channels, out_channels)

        # Match dimensions for residual addition
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, bias=False) \
                        if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out + self.shortcut(x)   # residual addition


# =============================================================
# BLOCK 3 — Stem Block (very first block, before residual)
# =============================================================

class StemBlock(nn.Module):
    """
    Entry block: Conv → BNReLUConv with residual shortcut.
    Handles raw pixel input (no BN before first conv).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, padding=1, bias=False)
        self.conv2    = BNReLUConv(out_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.shortcut(x)


# =============================================================
# BLOCK 4 — Encoder Stage (downsample + ResBlock)
# =============================================================

class EncoderStage(nn.Module):
    """Strided conv downsample → Residual block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=2, stride=2, bias=False)
        self.res_block  = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        return self.res_block(x)


# =============================================================
# BLOCK 5 — Decoder Stage (upsample + skip + ResBlock)
# =============================================================

class DecoderStage(nn.Module):
    """Bilinear upsample → concat skip → Residual block"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample  = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)


# =============================================================
# MAIN MODEL — ResUNet
# =============================================================

class ResUNet(nn.Module):
    """
    Residual U-Net for binary segmentation.

    Args:
        n_channels (int): Input image channels — use 3 for RGB CT slices
        n_classes  (int): Output channels     — use 1 for binary segmentation

    Forward pass returns raw logits.
    Apply torch.sigmoid() at inference time.
    Loss function should be BCEWithLogitsLoss (handles sigmoid internally).
    """

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.stem   = StemBlock(n_channels, 64)         # [B,  64, H,    W   ]
        self.enc2   = EncoderStage(64,  128)             # [B, 128, H/2,  W/2 ]
        self.enc3   = EncoderStage(128, 256)             # [B, 256, H/4,  W/4 ]
        self.enc4   = EncoderStage(256, 512)             # [B, 512, H/8,  W/8 ]

        # ── Bottleneck ───────────────────────────────────────
        self.bridge = EncoderStage(512, 1024)            # [B,1024, H/16, W/16]

        # ── Decoder ──────────────────────────────────────────
        self.dec4   = DecoderStage(1024, 512, 512)       # [B, 512, H/8,  W/8 ]
        self.dec3   = DecoderStage(512,  256, 256)       # [B, 256, H/4,  W/4 ]
        self.dec2   = DecoderStage(256,  128, 128)       # [B, 128, H/2,  W/2 ]
        self.dec1   = DecoderStage(128,  64,  64)        # [B,  64, H,    W   ]

        # ── Output head ──────────────────────────────────────
        self.head   = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)      # raw logits, no sigmoid
        )

    def forward(self, x):
        # Encoder — save skip connections
        s1 = self.stem(x)     # skip 1
        s2 = self.enc2(s1)    # skip 2
        s3 = self.enc3(s2)    # skip 3
        s4 = self.enc4(s3)    # skip 4

        # Bottleneck
        x  = self.bridge(s4)

        # Decoder — fuse with skips
        x  = self.dec4(x,  s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)

        return self.head(x)   # raw logits


# =============================================================
# QUICK TEST — run: python model.py
# =============================================================

if __name__ == "__main__":
    model = ResUNet(n_channels=3, n_classes=1)
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)

    print("=" * 45)
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Total params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 45)
    print("  ResUNet working correctly!")
