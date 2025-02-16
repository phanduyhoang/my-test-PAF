import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# These are your custom modules.
from opts.viz_opts import VizOpts
from data_process.coco import CocoDataSet

# --------------------------------------------------
# Dataset wrapping the COCO dataset
# --------------------------------------------------
class CocoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, opts, split='train'):
        self.coco_dataset = CocoDataSet(data_path, opts, split)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, heatmaps, paf, ignore_mask, keypoints = self.coco_dataset.get_item_raw(idx)
        # Convert to tensors; adjust dimensions if needed.
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        return img, heatmaps

# --------------------------------------------------
# Keypoint Head (VGG version)
# --------------------------------------------------
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

# --------------------------------------------------
# Complete Keypoint Model with fine-tuning on part of the backbone
# --------------------------------------------------
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19, fine_tune_backbone=True):
        super(KeypointModel, self).__init__()
        # Use a pretrained VGG19 and take a subset of layers.
        vgg = models.vgg19(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
        self.head = KeypointHead(in_channels=128, num_keypoints=num_keypoints)
        
        if fine_tune_backbone:
            # Freeze the EARLIER layers to preserve generic features.
            # Let the later layers be trainable to quickly adapt to our keypoint task.
            for idx, param in enumerate(self.backbone.parameters()):
                # Freeze roughly the first 60% of the parameters.
                if idx < int(0.6 * len(list(self.backbone.parameters()))):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # Freeze entire backbone.
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps

# --------------------------------------------------
# Helper: Save checkpoint function
# --------------------------------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

# --------------------------------------------------
# Training and Validation Pipeline
# --------------------------------------------------
def train_and_validate():
    # Set seeds for reproducibility.
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Parse options and set data path.
    opts = VizOpts().parse()
    data_path = opts.data

    # Create datasets.
    train_dataset = CocoTrainDataset(data_path, opts, split='train')
    val_dataset = CocoTrainDataset(data_path, opts, split='val')

    # Increase batch size and DataLoader efficiency.
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Initialize the model with partial backbone fine-tuning.
    model = KeypointModel(num_keypoints=19, fine_tune_backbone=True).to(device)

    # Use different learning rates for backbone (lower) and head (higher).
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    # CosineAnnealingWarmRestarts for a smoother LR schedule.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    num_epochs = 3000

    # Initialize GradScaler for mixed precision.
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for imgs, gt_heatmaps in pbar:
            imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, gt_heatmaps)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, gt_heatmaps in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    loss = criterion(preds, gt_heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        save_checkpoint(checkpoint, 'latest_checkpoint.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint, 'best_model.pth')
            print(f"New best model at epoch {epoch+1} with val loss {val_loss:.6f}.")

        scheduler.step()
        torch.cuda.empty_cache()

    # Save loss history for analysis.
    np.save('train_loss.npy', np.array(train_losses))
    np.save('val_loss.npy', np.array(val_losses))
    print("Training complete. Best model saved as 'best_model.pth'. Loss history saved.")

    visualize_predictions(model, val_loader, device)

# --------------------------------------------------
# Visualization: Show sample predictions vs. ground truth.
# --------------------------------------------------
def visualize_predictions(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        sample_img, sample_gt_heatmaps = next(iter(val_loader))
        sample_img = sample_img.to(device)
        with torch.cuda.amp.autocast():
            pred_heatmaps = model(sample_img)
        pred_heatmaps_np = pred_heatmaps.cpu().numpy()
        gt_heatmaps_np = sample_gt_heatmaps.numpy()

        # Display the first image in the batch.
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
    train_and_validate()
