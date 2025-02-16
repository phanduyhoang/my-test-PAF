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
from data_process.coco1 import CocoDataSet

# --------------------------------------------------
# Wrap the COCO dataset in a PyTorch Dataset
# --------------------------------------------------
class CocoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, opts, split='train'):
        self.coco_dataset = CocoDataSet(data_path, opts, split)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, heatmaps, paf, ignore_mask, keypoints = self.coco_dataset.get_item_raw(idx)
        # Convert to tensors and adjust dimensions if needed.
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        return img, heatmaps

# --------------------------------------------------
# Define a Grouped Keypoint Head where each branch is dedicated to one keypoint.
# --------------------------------------------------
class GroupedKeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints, branch_channels=256):
        """
        Args:
            in_channels (int): Number of input channels from the backbone.
            num_keypoints (int): Number of keypoints (each branch outputs one channel).
            branch_channels (int): Number of channels in the branch layers.
        """
        super(GroupedKeypointHead, self).__init__()
        self.num_keypoints = num_keypoints
        
        # Each branch is a small network that upsamples the features and outputs one heatmap.
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(branch_channels, 1, kernel_size=1)
            )
            for _ in range(num_keypoints)
        ])

    def forward(self, x):
        # Each branch produces an output heatmap (batch_size, 1, H, W)
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along the channel dimension to form (batch_size, num_keypoints, H, W)
        heatmaps = torch.cat(branch_outputs, dim=1)
        return heatmaps

# --------------------------------------------------
# Define the complete Keypoint Model (VGG backbone)
# --------------------------------------------------
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19):
        super(KeypointModel, self).__init__()
        # Use a pretrained VGG19 and take a subset of layers.
        vgg = models.vgg19(pretrained=True)
        # Here we use more layers if needed; adjust the slice if you want to expose more spatial resolution.
        self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
        # Replace the simple head with the grouped head.
        self.head = GroupedKeypointHead(in_channels=128, num_keypoints=num_keypoints, branch_channels=256)

    def forward(self, x):
        # Freeze the backbone during training to focus on the head.
        with torch.no_grad():
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
    # Set seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Parse options and set data path
    opts = VizOpts().parse()
    data_path = opts.data

    # Create datasets for training and validation splits
    train_dataset = CocoTrainDataset(data_path, opts, split='train')
    val_dataset = CocoTrainDataset(data_path, opts, split='val')

    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Initialize the model and move it to the device.
    model = KeypointModel(num_keypoints=19).to(device)

    # Only the head parameters are optimized.
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    num_epochs = 3000

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Create a tqdm progress bar for the training loader.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for imgs, gt_heatmaps in pbar:
            imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, gt_heatmaps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            # Update tqdm with the current loss.
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, gt_heatmaps in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
                preds = model(imgs)
                loss = criterion(preds, gt_heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Create the checkpoint dictionary
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        
        # Save the latest checkpoint (overwrites the previous one)
        save_checkpoint(checkpoint, 'latest_checkpoint.pth')

        # Save the best model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint, 'best_model.pth')
            print(f"New best model found at epoch {epoch+1} with val loss {val_loss:.6f}.")

        scheduler.step()
        torch.cuda.empty_cache()

    # Save loss history for future analysis.
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
    train_and_validate()
