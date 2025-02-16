import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
# from torchvision.models import mobilenet_v3_small  # Import MobileNet V3 Small

from opts.viz_opts import VizOpts
from data_process.coco import CocoDataSet
from tqdm import tqdm

# --------------------------------------------------
# Wrap your custom CocoDataSet in a PyTorch Dataset
# --------------------------------------------------
class CocoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, opts, split='train'):
        self.coco_dataset = CocoDataSet(data_path, opts, split)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, heatmaps, paf, ignore_mask, keypoints = self.coco_dataset.get_item_raw(idx)
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        if not torch.is_tensor(heatmaps):
            heatmaps = torch.from_numpy(heatmaps).float()
        return img, heatmaps

# --------------------------------------------------
# Define the Keypoint Head (same for both backbones)
# --------------------------------------------------
# This is for VGG
class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointHead, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        return x

# Alternative Keypoint Head for MobileNet (for reference; do not uncomment unless using MobileNet)
# class KeypointHead(nn.Module):
#     def __init__(self, in_channels, num_keypoints):
#         super(KeypointHead, self).__init__()
#         # For VGG backbone (old code), you had two upsampling layers:
#         # For MobileNet, we need more upsampling to match the GT heatmap size.
#         # Assuming the backbone outputs a feature map with spatial size ≈23×23,
#         # we need a total upsampling factor of 16 (2^4 = 16) to reach ~368×368.
#         self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv4 = nn.Conv2d(256, num_keypoints, kernel_size=1)
#
#     def forward(self, x):
#         x = self.upsample1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x = self.upsample2(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         x = self.upsample3(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#
#         x = self.upsample4(x)
#         x = self.conv4(x)
#         return x

# --------------------------------------------------
# Define the Keypoint Model with VGG backbone (MobileNet alternative commented)
# --------------------------------------------------
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19):
        super(KeypointModel, self).__init__()
        
        # VGG Code:
        vgg = models.vgg19(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # NEW MobileNet V3 Small Backbone (commented out):
        # feat = mobilenet_v3_small(weights='DEFAULT')
        # Here we take a subset of layers. Adjust the slice as needed.
        # self.backbone = nn.Sequential(*list(feat.features.children())[:-5]).eval()
        
        # OLD Head for VGG backbone (in_channels=128):
        self.head = KeypointHead(in_channels=128, num_keypoints=num_keypoints)
        
        # NEW Head for MobileNet backbone (in_channels=48):
        # self.head = KeypointHead(in_channels=48, num_keypoints=num_keypoints)

    def forward(self, x):
        # Freeze the backbone by wrapping in torch.no_grad()
        with torch.no_grad():
            features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps

# --------------------------------------------------
# Training and evaluation pipeline
# --------------------------------------------------
def train_and_visualize():
    # Set seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    opts = VizOpts().parse()
    data_path = opts.data

    # Use full COCO training and validation datasets
    train_dataset = CocoTrainDataset(data_path, opts, split='train')
    val_dataset = CocoTrainDataset(data_path, opts, split='val')
    
    # Adjust the batch size to help limit GPU memory usage
    batch_size = 4  # Lower batch size uses less GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Set the device and attempt to limit GPU memory usage if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        try:
            # Limit per-process GPU memory to 50% of the available memory (if supported)
            torch.cuda.set_per_process_memory_fraction(0.5, device=device)
            print("Set GPU memory fraction to 0.5")
        except Exception as e:
            print("Could not set GPU memory fraction:", e)
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    model = KeypointModel(num_keypoints=19).to(device)
    
    # Only the head's parameters are being optimized
    optimizer = optim.Adam(model.head.parameters(), lr=1e-2, weight_decay=1e-4)
    
    num_epochs = 150
    steps_per_epoch = len(train_loader)
    # OneCycleLR scheduler: ramps the LR up to max_lr and then down, preventing it from dropping to 0.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs
    )
    
    criterion = nn.MSELoss()
    train_loss_values = []
    val_loss_values = []

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        epoch_train_loss = 0.0
        for imgs, gt_heatmaps in (train_loader):
            imgs = imgs.to(device)
            gt_heatmaps = gt_heatmaps.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, gt_heatmaps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent NaNs
            optimizer.step()
            scheduler.step()  # Step the scheduler per batch

            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_loss_values.append(epoch_train_loss)

        # ----- Evaluation -----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for imgs, gt_heatmaps in val_loader:
                imgs = imgs.to(device)
                gt_heatmaps = gt_heatmaps.to(device)
                preds = model(imgs)
                loss = criterion(preds, gt_heatmaps)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        val_loss_values.append(epoch_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, LR: {current_lr:.6f}")

        if epoch_train_loss < 1e-6:
            print("Train Loss is very low; stopping early.")
            break

    torch.save(model, 'full_model.pth')
    print("Full model saved as 'full_model.pth'")
    np.save('train_loss_values.npy', np.array(train_loss_values))
    np.save('val_loss_values.npy', np.array(val_loss_values))
    print("Loss values saved as 'train_loss_values.npy' and 'val_loss_values.npy'")

    # ----- Visualization -----
    # Show one sample from the validation set
    model.eval()
    with torch.no_grad():
        sample_img, sample_gt_heatmaps = next(iter(val_loader))
        sample_img = sample_img.to(device)
        pred_heatmaps = model(sample_img)
        pred_heatmaps_np = pred_heatmaps.cpu().numpy()
        gt_heatmaps_np = sample_gt_heatmaps.numpy()

        # Display the first image in the batch
        sample_img_np = sample_img[0].cpu().permute(1, 2, 0).numpy()
        sample_img_np = np.clip(sample_img_np, 0, 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(sample_img_np)
        plt.title("Input Image")
        plt.axis("off")
        plt.show()

        num_keypoints = pred_heatmaps_np.shape[1]
        fig, axs = plt.subplots(num_keypoints, 2, figsize=(8, num_keypoints * 2))
        for i in range(num_keypoints):
            axs[i, 0].imshow(gt_heatmaps_np[0, i], cmap="hot")
            axs[i, 0].set_title(f"GT Channel {i}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(pred_heatmaps_np[0, i], cmap="hot")
            axs[i, 1].set_title(f"Predicted Channel {i}")
            axs[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    train_and_visualize()
