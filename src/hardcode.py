import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ---------------------------
# CONFIGURATION (update paths)
# ---------------------------
model_path = "model_60.pth"  
# Set the image path to a valid image from your dataset:
image_path = "kaggle\working\coco2017\train2017\000000259099.jpg"  # <<-- UPDATE THIS

# ---------------------------
# CREATE THE MODEL ARCHITECTURE
# ---------------------------
from opts.base_opts import Opts
from model.model_provider import create_model  # This should create your architecture

# Create a dummy options object for instantiation (update as necessary)
opt = Opts().parse()

# Create model architecture
model, _, _ = create_model(opt)

# Load the state_dict from your file (assuming it's a state_dict)
state_dict = torch.load(model_path, map_location='cuda')
model.load_state_dict(state_dict)

# Move model to GPU and set to evaluation mode
model = model.cuda()
model.eval()

# ---------------------------
# READ THE IMAGE
# ---------------------------
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise ValueError(f"Image not found at: {image_path}")
# Convert BGR to RGB for display with matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Normalize image to [0,1] as float32 (adjust if your model expects different preprocessing)
img_rgb = img_rgb.astype(np.float32) / 255.0

# Convert image to tensor: shape [1, 3, H, W]
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda()

# ---------------------------
# RUN INFERENCE
# ---------------------------
with torch.no_grad():
    output = model(img_tensor)
    
# If model returns multiple outputs (multi-stage), take the last one.
if isinstance(output, list):
    output = output[-1]

# Remove the batch dimension: now shape is [C_out, H_out, W_out]
output = output.cpu().squeeze(0)

# ---------------------------
# UPSAMPLE TO IMAGE RESOLUTION (if necessary)
# ---------------------------
H_img, W_img, _ = img_rgb.shape
if output.shape[1:] != (H_img, W_img):
    output_vis = F.interpolate(output.unsqueeze(0), size=(H_img, W_img),
                               mode='bilinear', align_corners=False).squeeze(0)
else:
    output_vis = output

# Convert to NumPy array
output_np = output_vis.numpy()  # shape: [C_out, H, W]
num_channels = output_np.shape[0]
print(f"Model output has {num_channels} channels.")

# ---------------------------
# DISPLAY ALL CHANNELS
# ---------------------------
cols = 4  # Number of columns in the grid
rows = (num_channels + cols - 1) // cols  # Compute number of rows needed

fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axs = axs.flatten()

for i in range(num_channels):
    axs[i].imshow(output_np[i], cmap='jet')
    axs[i].set_title(f"Channel {i}")
    axs[i].axis("off")

# Hide extra subplots if any
for j in range(num_channels, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
