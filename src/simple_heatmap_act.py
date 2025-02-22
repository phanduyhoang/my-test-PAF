import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.models import mobilenet_v3_small  # Import MobileNet V3 Small

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
        img, heatmaps, paf, ignore_mask, keypoints = self.coco_dataset.get_item_raw(idx)
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        if not torch.is_tensor(heatmaps):
            heatmaps = torch.from_numpy(heatmaps).float()
        return img, heatmaps

# --------------------------------------------------
# Define the Keypoint Head (same for both backbones)
# --------------------------------------------------

#this is for vgg
# class KeypointHead(nn.Module):
#     def __init__(self, in_channels, num_keypoints):
#         super(KeypointHead, self).__init__()
#         self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)

#     def forward(self, x):
#         x = self.upsample1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.upsample2(x)
#         x = self.conv2(x)
#         return x





class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointHead, self).__init__()
        # For VGG backbone (old code), you had two upsampling layers:
        # self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)
        #
        # For MobileNet, we need more upsampling to match the GT heatmap size.
        # Assuming the backbone outputs a feature map with spatial size ≈23×23,
        # we need a total upsampling factor of 16 (2^4 = 16) to reach ~368×368.

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(256, num_keypoints, kernel_size=1)
        self.act=nn.ReLU()

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.upsample4(x)
        x = self.conv4(x)
        x=self.act(x)        
        return x


# --------------------------------------------------
# Define the Keypoint Model with MobileNet V3 Small backbone
# --------------------------------------------------
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19):
        super(KeypointModel, self).__init__()
        
        # VGG Code:
        # vgg = models.vgg19(pretrained=True)
        # self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # NEW MobileNet V3 Small Backbone:
        feat = mobilenet_v3_small(weights='DEFAULT')
        # Here we take a subset of layers. Adjust the slice as needed.
        self.backbone = nn.Sequential(*list(feat.features.children())[:-5]).eval()
        
        # OLD Head for VGG backbone (in_channels=128):
        # self.head = KeypointHead(in_channels=128, num_keypoints=num_keypoints)
        
        # NEW Head for MobileNet backbone (in_channels=48):
        self.head = KeypointHead(in_channels=48, num_keypoints=num_keypoints)

    def forward(self, x):
        # Freeze the backbone by wrapping in torch.no_grad()
        with torch.no_grad():
            features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps

# --------------------------------------------------
# Training and visualization pipeline
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

    full_dataset = CocoTrainDataset(data_path, opts, split='train')
    subset_indices = list(range(128))# Overfit on one image[:127]
    train_subset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=1)
    #train_loader = DataLoader(subset_indices, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = KeypointModel(num_keypoints=19).to(device)
    
    # Only the head's parameters are being optimized
    optimizer = optim.Adam(model.head.parameters(), lr=1e-2, weight_decay=1e-4)
    
    num_epochs = 4000
    steps_per_epoch = len(train_loader)  # Likely 1 since we're overfitting on one image

    # OneCycleLR scheduler: ramps the LR up to max_lr and then down, preventing it from dropping to 0.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs
    )

    criterion = nn.MSELoss()
    loss_values = []
    model.train()
    for epoch in range(num_epochs): 
            epoch_loss = 0.0 
            for imgs, gt_heatmaps in (train_loader): 
                imgs = imgs.to(device) 
                gt_heatmaps = gt_heatmaps.to(device) 
            
                
                # for imgs, gt_heatmaps in train_loader: 
                #     imgs = imgs.to(device) 
                #     gt_heatmaps = gt_heatmaps.to(device) 
    
                optimizer.zero_grad() 
                preds = model(imgs) 
                loss = criterion(preds, gt_heatmaps) 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent NaNs 
                optimizer.step() 
                scheduler.step()  # Step the scheduler per batch 
    
                epoch_loss += loss.item() 
    
    
            epoch_loss /= len(train_loader) 
            loss_values.append(epoch_loss) 
            current_lr = optimizer.param_groups[0]['lr'] 
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}") 
    
            if epoch_loss < 5e-8: 
                print("Loss is very low; stopping early.") 
                break 
    
    torch.save(model, 'full_model.pth') 
    print("Full model saved as 'full_model.pth'") 
    np.save('loss_values.npy', np.array(loss_values)) 
    print("Loss values saved as 'loss_values.npy'")


    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        sample_img, sample_gt_heatmaps = train_subset[0]
        sample_img = sample_img.to(device)
        pred_heatmaps = model(sample_img.unsqueeze(0))
        pred_heatmaps_np = pred_heatmaps.cpu().squeeze(0).numpy()
        gt_heatmaps_np = sample_gt_heatmaps.numpy()

        sample_img_np = sample_img.cpu().permute(1, 2, 0).numpy()
        sample_img_np = np.clip(sample_img_np, 0, 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(sample_img_np)
        plt.title("Input Image")
        plt.axis("off")
        plt.show()

        fig, axs = plt.subplots(19, 2, figsize=(8, 38))
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
