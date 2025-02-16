import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import models

from opts.viz_opts import VizOpts
from data_process.coco import CocoDataSet

# --------------------------------------------------
# Wrap your custom CocoDataSet in a PyTorch Dataset
# --------------------------------------------------
class CocoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, opts, split='train'):
        self.coco_dataset = CocoDataSet(data_path, opts, split)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        # get_item_raw returns: (img, heatmaps, paf, ignore_mask, keypoints)
        img, heatmaps, paf, ignore_mask, keypoints = self.coco_dataset.get_item_raw(idx)
        # Convert image to CHW tensor of type float32.
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img = img.float()
        # Convert heatmaps to tensor of type float32.
        if not torch.is_tensor(heatmaps):
            heatmaps = torch.from_numpy(heatmaps).float()
        else:
            heatmaps = heatmaps.float()
        return img, heatmaps

# --------------------------------------------------
# Define the model components
# --------------------------------------------------
class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointHead, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu1(self.conv1(x))
        x = self.upsample2(x)
        x = self.conv2(x)
        return x

class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19):
        super(KeypointModel, self).__init__()
        vgg = models.vgg19(pretrained=True)
        # Use the first 10 layers of VGG19 as backbone.
        self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
        self.head = KeypointHead(in_channels=128, num_keypoints=num_keypoints)

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps

# --------------------------------------------------
# Training and visualization pipeline
# --------------------------------------------------
def train_and_visualize():
    # Set random seeds for reproducibility.
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Parse options using your custom VizOpts.
    opts = VizOpts().parse()
    data_path = opts.data

    # Create training dataset using your custom COCO dataset.
    full_dataset = CocoTrainDataset(data_path, opts, split='train')
    # For absolute overfitting, use only one image (index 0).
    subset_indices = [0]
    train_subset = Subset(full_dataset, subset_indices)
    # Use batch size of 1.
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = KeypointModel(num_keypoints=19).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Set up mixed precision training (optional).
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 10000  # Increase epochs to ensure complete overfitting.
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, gt_heatmaps in train_loader:
            imgs = imgs.to(device)
            gt_heatmaps = gt_heatmaps.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, gt_heatmaps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_subset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        torch.cuda.empty_cache()

    # Save the full model (architecture + parameters).
    torch.save(model, 'full_model.pth')
    print("Full model saved as 'full_model.pth'")

    # --------------------------------------------------
    # Visualization: Display input image and all 19 keypoint channels.
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Get the single sample from the training subset.
        sample_img, sample_gt_heatmaps = train_subset[0]
        sample_img = sample_img.to(device)
        pred_heatmaps = model(sample_img.unsqueeze(0))
        # Move to CPU and convert to numpy arrays.
        pred_heatmaps_np = pred_heatmaps.cpu().squeeze(0).numpy()  # Shape: (19, H, W)
        gt_heatmaps_np = sample_gt_heatmaps.numpy()                 # Shape: (19, H, W)

        # Display the input image.
        sample_img_np = sample_img.cpu().permute(1, 2, 0).numpy()
        sample_img_np = np.clip(sample_img_np, 0, 1)
        plt.figure(figsize=(6,6))
        plt.imshow(sample_img_np)
        plt.title("Input Image")
        plt.axis("off")
        plt.show()

        # Create a figure that shows all 19 channels.
        fig, axs = plt.subplots(19, 2, figsize=(8, 38))  # 19 rows, 2 columns.
        for i in range(19):
            axs[i, 0].imshow(gt_heatmaps_np[i], cmap="hot")
            axs[i, 0].set_title(f"GT Channel {i}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(pred_heatmaps_np[i], cmap="hot")
            axs[i, 1].set_title(f"Predicted Channel {i}")
            axs[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    train_and_visualize()
