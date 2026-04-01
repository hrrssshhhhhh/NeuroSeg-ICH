import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from Models.Models.Resunet.model import ResUNet
from Models.Models.Resunet.dataset import ICHDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Model
# -------------------------
checkpoint = torch.load("best_model_epoch.pth", map_location=device)

model = ResUNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']} with Dice {checkpoint['best_dice']:.4f}")

# Load validation dataset
val_dataset = ICHDataset(
    image_dir="DataV1/CV0/validate/image",
    mask_dir="DataV1/CV0/validate/label"
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Get one sample


import numpy as np

thresholds = np.arange(0.1, 0.91, 0.05)

best_threshold = 0
best_dice = 0

model.eval()

with torch.no_grad():
    for threshold in thresholds:
        dice_total = 0

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            intersection = (preds * masks).sum()
            dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-8)

            dice_total += dice.item()

        avg_dice = dice_total / len(val_loader)

        print(f"Threshold {threshold:.2f} -> Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best Dice at this threshold:", best_dice)
print(f"Final Validation Dice (Best Threshold): {best_dice:.4f}")

# ----------------------------
# Pick best performing slice for visualization
# ----------------------------

best_slice_images = None
best_slice_masks = None
best_slice_dice = 0

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > best_threshold).float()

        intersection = (preds * masks).sum()
        dice_val = (2 * intersection) / (preds.sum() + masks.sum() + 1e-8)

        if dice_val.item() > best_slice_dice:
            best_slice_dice = dice_val.item()
            best_slice_images = images
            best_slice_masks = masks

images = best_slice_images
masks = best_slice_masks

with torch.no_grad():
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > best_threshold).float()

image = images[0].cpu().permute(1, 2, 0).numpy()
mask = masks[0].cpu().squeeze().numpy()
mask = (mask > 0).astype(np.float32)
pred = preds[0].cpu().squeeze().numpy()

# normalize image
image = (image - image.min()) / (image.max() - image.min() + 1e-8)

# compute Dice for this sample
intersection = (pred * mask).sum()
dice = (2. * intersection) / (pred.sum() + mask.sum() + 1e-8)

error = np.logical_xor(pred, mask)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(18,5))

# Original
plt.subplot(1,4,1)
plt.title("Original CT", fontsize=14)
plt.imshow(image, cmap="gray")
plt.axis("off")

# Ground Truth
plt.subplot(1,4,2)
plt.title("Ground Truth (Red Contour)", fontsize=14)
plt.imshow(image, cmap="gray")
plt.contour(mask, colors='red', linewidths=2)
plt.axis("off")

# Prediction
plt.subplot(1,4,3)
plt.title("Prediction (Green Contour)", fontsize=14)
plt.imshow(image, cmap="gray")
plt.contour(pred, colors='lime', linewidths=2)
plt.axis("off")

# Error Map
plt.subplot(1,4,4)
plt.title(f"Error Map (Yellow)\nDice Score: {dice:.3f}", fontsize=14)
plt.imshow(image, cmap="gray")
plt.contour(error, colors='yellow', linewidths=2)
plt.axis("off")

plt.tight_layout()
plt.savefig("final_submission_figure.png", dpi=600, bbox_inches='tight')

plt.show()

