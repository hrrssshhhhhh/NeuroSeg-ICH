import sys
sys.path.append("Models/Models/Resunet")

from dataset import ICHDataset
from torch.utils.data import DataLoader

train_dataset = ICHDataset(
    image_dir="DataV1/CV0/train/image",
    mask_dir="DataV1/CV0/train/label"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print("Total training samples:", len(train_dataset))

for images, masks in train_loader:
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    break