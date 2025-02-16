import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model, create_optimizer
from training.train_net import validate_net

def main():
    # Seed all sources of randomness for reproducibility.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    opt = Opts().parse()

    # Create data loaders (using test loader)
    _, test_loader = create_data_loaders(opt)

    # Create model and loss criteria
    model, criterion_hm, criterion_paf = create_model(opt)
    model = model.cuda()
    criterion_hm = criterion_hm.cuda()
    criterion_paf = criterion_paf.cuda()

    # Create optimizer if needed (not used in evaluation, but kept for consistency)
    optimizer = create_optimizer(opt, model)

    # Load the saved model weights for evaluation only
    if hasattr(opt, 'loadModel') and opt.loadModel:
        print(f"Loading model from {opt.loadModel}")
        checkpoint = torch.load(opt.loadModel)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            # The checkpoint is an entire model
            model = checkpoint.cuda()
    else:
        print("No pre-trained model specified via -loadModel. Exiting.")
        return

    # Set model to evaluation mode
    model.eval()
    torch.cuda.empty_cache()

    # Display combined predicted heatmaps for 4 images from the test set (without overlaying original images)
    display_predicted_heatmaps(test_loader, model, num_images=4)

def display_predicted_heatmaps(data_loader, model, num_images=4):
    """
    Selects num_images from the first batch of data_loader, runs inference, 
    combines the predicted heatmap channels, normalizes the result,
    and displays only the combined predicted heatmaps.
    """
    # Get one batch (assume batch[0] contains input images)
    for batch in data_loader:
        images = batch[0]  # shape: [B, C, H, W]
        break

    B = images.size(0)
    num_to_display = min(num_images, B)
    print(f"Displaying combined predicted heatmaps for {num_to_display} images.")

    for i in range(num_to_display):
        img = images[i].unsqueeze(0).float().cuda()  # [1, C, H, W]
        with torch.no_grad():
            pred_heatmaps, _ = model(img)
        if isinstance(pred_heatmaps, list):
            pred_heatmaps = pred_heatmaps[-1]
        pred_heatmaps = pred_heatmaps.cpu().squeeze(0)  # [num_channels, H_pred, W_pred]

        # Upsample to match input image resolution if necessary.
        _, H, W = img.shape[1:]
        if pred_heatmaps.shape[1:] != (H, W):
            pred_heatmaps = F.interpolate(pred_heatmaps.unsqueeze(0),
                                          size=(H, W),
                                          mode='bilinear', align_corners=False).squeeze(0)

        # Convert predicted heatmaps to numpy array.
        pred_heatmaps_np = pred_heatmaps.numpy()  # shape: [num_channels, H, W]

        # Combine all channels by summing and then normalize.
        combined_heatmap = np.sum(pred_heatmaps_np, axis=0)
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)

        # Debug print
        print(f"Image {i+1}: Combined Heatmap stats - min: {combined_heatmap.min():.4f}, max: {combined_heatmap.max():.4f}, mean: {combined_heatmap.mean():.4f}")

        # Display the combined predicted heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(combined_heatmap, cmap='jet')
        plt.title(f"Combined Predicted Heatmap for Image {i+1}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
