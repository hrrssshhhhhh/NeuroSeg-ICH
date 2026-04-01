import sys
sys.path.append("Models/Models/Resunet")

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import ICHDataset
from model import ResUNet   # make sure model file name matches
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset
# -------------------------
train_dataset = ICHDataset(
    image_dir="DataV1/CV0/train/image",
    mask_dir="DataV1/CV0/train/label"
)

val_dataset = ICHDataset(
    image_dir="DataV1/CV0/validate/image",
    mask_dir="DataV1/CV0/validate/label"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# -------------------------
# Model
# -------------------------
model = ResUNet(n_channels=3, n_classes=1).to(device)

# -------------------------
# Loss (BCE + Dice)
# -------------------------
pos_weight = torch.tensor([20.0]).to(device)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # because we monitor Dice (higher is better)
    factor=0.5,        # reduce LR by half
    patience=5,        # wait 5 epochs without improvement
    
)
def dice_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

def precision_recall(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)

    return precision, recall
# -------------------------
# Training Loop
# -------------------------
# Metric storage for training curve visualization
epochs = 50
best_dice = 0
train_losses = []
val_losses = []
val_dices = []
val_ious = []
val_precisions = []
val_recalls = []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    dice_total = 0
    iou_total = 0
    precision_total = 0
    recall_total = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            val_loss += loss.item()
            dice_total += dice_score(outputs, masks).item()
            iou_total += iou_score(outputs, masks).item()
            precision, recall = precision_recall(outputs, masks)
            precision_total += precision.item()
            recall_total += recall.item()

    val_loss_avg = val_loss / len(val_loader)
    val_dice = dice_total / len(val_loader)
    val_iou = iou_total / len(val_loader)
    val_precision = precision_total / len(val_loader)
    val_recall = recall_total / len(val_loader)
# ----------------------------
# Log epoch metrics for later plotting
# ----------------------------
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss_avg)
    val_dices.append(val_dice)
    val_ious.append(val_iou)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)

    print(f"Validation Loss: {val_loss_avg:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    scheduler.step(val_dice)
    print(f"Validation IoU: {val_iou:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save({
          "epoch": epoch + 1,
          "model_state_dict": model.state_dict(),
          "best_dice": best_dice
        }, "best_model_epoch.pth")

    print(f"Best model saved at epoch {epoch+1} with Dice {best_dice:.4f}")

print("Training Complete.");

# ----------------------------
# Plot and save training curves (Loss & Evaluation Metrics)
# ----------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(val_dices, label="Val Dice")
plt.plot(val_ious, label="Val IoU")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Curves")
plt.legend()
plt.grid(True)

plt.savefig("training_curves.png", dpi=300)
plt.show()