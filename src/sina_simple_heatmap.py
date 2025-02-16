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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
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
# class KeypointHead(nn.Module):
#     def __init__(self, in_channels, num_keypoints):
#         super(KeypointHead, self).__init__()
#         self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, padding=0)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)

#     def forward(self, x):
#         x = self.upsample1(x)
#         # print("SS", x.shape)
#         x = self.relu1(self.conv1(x))
#         # print("SSS", x.shape)
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
        return x



# class KeypointModel(nn.Module):
#     def __init__(self, num_keypoints=19):
#         super(KeypointModel, self).__init__()
#         vgg = models.vgg19(pretrained=True)
#         # Use the first 10 layers of VGG19 as backbone.
#         self.backbone = nn.Sequential(*list(vgg.features.children())[:10])
#         self.head = KeypointHead(in_channels=128, num_keypoints=num_keypoints)

#     def forward(self, x):
#         features = self.backbone(x)
#         # print(features.shape)
#         heatmaps = self.head(features)
#         return heatmaps

class BBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 1, skip=True):
        super(BBlock, self).__init__()

        self.skip_flag = skip
        self.core = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=groups),
        )

        self.skp = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        bb = self.core(x)

        if self.skip_flag:
            sk = self.skp(x)
            return bb + sk
        
        return bb

class DCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 1, skip = True):
        super(DCBlock, self).__init__()

        self.skip_flag = skip
        self.core = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups),
            BBlock(out_channels, out_channels, groups=groups),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.skp = BBlock(in_channels, out_channels, groups=groups, skip = skip)

    def forward(self, x):
        
        dc = self.core(x)

        if self.skip_flag:
            sk = self.skp(x)
            return dc + sk
        
        return dc 
    
    

class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=19):
        super(KeypointModel, self).__init__()

        # skip = True
        # self.feat = torchvision.get_model("quantized_mobilenet_v3_large", weights="DEFAULT")
        feat = mobilenet_v3_small(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(feat.features.children())[:-5]).eval()
        # print(self.backbone)

        # self.head = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     BBlock(48, 32, skip=skip, groups=16),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     BBlock(32, 32, skip=skip, groups=4),
        #     BBlock(32, 19, skip=False, groups=1)

        #     # DCBlock(3, 255, skip=skip, groups=3),
        #     # DCBlock(255, 128, skip=skip, groups=1),
        #     # DCBlock(128, 64, skip=skip, groups=32),
        #     # DCBlock(64, 32, skip=skip, groups=16),
        #     # DCBlock(32, 19, skip=skip, groups=1),
        #     # BBlock(19, 19, skip=skip, groups=19),
        #     # nn.ReLU()
        # )
        self.head = KeypointHead(in_channels=48, num_keypoints=num_keypoints)



    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            # print(x.shape)
        heatmaps = self.head(x)
        return heatmaps

class FocalLoss(nn.Module):
    def __init__(self, main_loss = nn.BCEWithLogitsLoss(reduction="none"), gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.main_loss = main_loss
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        loss = self.main_loss(pred, target)

        pred_prob = pred.sigmoid()  # prob from logits
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)

        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * modulating_factor
        if self.alpha > 0:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = loss * alpha_factor
        return loss.mean(1).sum()
    
# --------------------------------------------------
# Training and visualization pipeline
# --------------------------------------------------
def train_and_visualize():
    # Set random seeds for reproducibility.
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(0)

    # Parse options using your custom VizOpts.
    opts = VizOpts().parse()
    data_path = opts.data

    # Create training dataset using your custom COCO dataset.
    full_dataset = CocoTrainDataset(data_path, opts, split='train')
    # For overfitting test, use only 32 images.
    subset_indices = list(range(1))
    train_subset = Subset(full_dataset, subset_indices)
    # Use a smaller batch size to ease GPU memory constraints.
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)

    device = torch.device('cuda')
    print("Using device:", device)

    model = KeypointModel(num_keypoints=19).to(device)

    total_params = sum(p.numel() for p in model.state_dict().values())
    print(f"Total parameters: {total_params}")
    

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    # Set up mixed precision training (optional).
    # scaler = torch.cuda.amp.GradScaler()

    num_epochs = 100  # You can increase epochs if needed.
    model.train()
    imgs, gt_heatmaps = next(iter(train_loader))
    imgs = imgs.to(device)
    gt_heatmaps = gt_heatmaps.to(device)
    
    print(gt_heatmaps.shape)



    for epoch in range(num_epochs):
        epoch_loss = []
        # for imgs, gt_heatmaps in train_loader:
        # imgs = imgs.to(device)
        # gt_heatmaps = gt_heatmaps.to(device)

        # gt_heatmaps = gt_heatmaps[:, 0, :, :].view(imgs.shape[0], 1, gt_heatmaps.shape[2], gt_heatmaps.shape[3])

        # print(torch.min(gt_heatmaps[:, -1, :, :]), torch.max(gt_heatmaps[:, -1, :, :]))
        #gt_heatmaps = torch.clamp(gt_heatmaps, min=0., max=1.) 

        
        # with torch.cuda.amp.autocast():
        optimizer.zero_grad()
        preds = model(imgs)

        
        # print(gt_heatmaps.shape, preds.shape)
        loss = criterion(preds, gt_heatmaps)
        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        epoch_loss.append(loss.item())

        # epoch_loss /= len(train_subset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.array(epoch_loss).mean():.7f}")
        # torch.cuda.empty_cache()

    # Save the full model (architecture + parameters).

    # steps_per_epoch = len(train_loader)
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-2,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=num_epochs
    # )
    # for epoch in range(num_epochs):
    #     epoch_loss = 0.0
    #     for imgs, gt_heatmaps in train_loader:
    #         imgs = imgs.to(device)
    #         gt_heatmaps = gt_heatmaps.to(device)

    #         optimizer.zero_grad()
    #         preds = model(imgs)
    #         loss = criterion(preds, gt_heatmaps)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent NaNs
    #         optimizer.step()
    #         scheduler.step()  # Step the scheduler per batch

    #         epoch_loss += loss.item()

    #     epoch_loss /= len(train_loader)
    #     #loss_values.append(epoch_loss)
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}")

    #     if epoch_loss < 1e-6:
    #         print("Loss is very low; stopping early.")
    #         break





    torch.save(model, 'full_model.pth')
    print("Full model saved as 'full_model.pth'")

    # --------------------------------------------------
    # Visualization: Display input image and all 19 keypoint channels.
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Get a sample from the training subset.
        sample_img, sample_gt_heatmaps = train_subset[0]

        # gt_heatmaps = gt_heatmaps[:, 0, :, :].view(imgs.shape[0], 1, gt_heatmaps.shape[2], gt_heatmaps.shape[3])

        sample_gt_heatmaps = torch.clamp(sample_gt_heatmaps, min=0., max=1.)

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
        n = 19
        fig, axs = plt.subplots(2, n, figsize=(20, 8))  # 19 rows, 2 columns.
        for i in range(n):
            axs[0, i].imshow(gt_heatmaps_np[i], cmap="hot")
            axs[0, i].set_title(f"GT_{i}")
            axs[0, i].axis("off")
            axs[1, i].imshow(pred_heatmaps_np[i], cmap="hot")
            axs[1, i].set_title(f"Pred_{i}")
            axs[1, i].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    train_and_visualize()
