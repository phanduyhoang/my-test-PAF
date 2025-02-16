import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model  # No need for create_optimizer in eval

def main():
    # Seed all sources of randomness for reproducibility.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    # Parse options (ensure -loadModel is set in your Opts)
    opt = Opts().parse()

    # Create dataloaders; now we use the train loader for overfitting.
    train_loader, _ = create_data_loaders(opt)

    # Get one batch from the train dataloader.
    # We assume:
    #   batch[0] -> input images [B, C, H, W]
    #   batch[1] -> GT heatmaps [B, num_hm_channels, H_gt, W_gt] (if available)
    #   batch[2] -> GT PAFs [B, num_paf_channels, H_paf, W_paf] (if available)
    for batch in train_loader:
        input_tensor = batch[0]
        gt_heatmaps = batch[1] if len(batch) > 1 else None
        gt_pafs     = batch[2] if len(batch) > 2 else None
        break  # use only the first batch

    # Randomly select an image from the batch.
    batch_size = input_tensor.size(0)
    random_index = random.randint(0, batch_size - 1)
    single_input = input_tensor[random_index].unsqueeze(0)  # [1, C, H, W]
    original_image_tensor = input_tensor[random_index]       # [C, H, W]

    if gt_heatmaps is not None:
        single_gt_heatmaps = gt_heatmaps[random_index]  # [num_hm_channels, H_gt, W_gt]
    else:
        single_gt_heatmaps = None

    if gt_pafs is not None:
        single_gt_pafs = gt_pafs[random_index]          # [num_paf_channels, H_paf, W_paf]
    else:
        single_gt_pafs = None

    single_input = single_input.float()

    # Create the model and obtain the loss functions.
    model, criterion_hm, criterion_paf = create_model(opt)
    model = model.cuda()

    # Load saved model weights or full model.
    if hasattr(opt, 'loadModel') and opt.loadModel:
        print(f"Loading model from {opt.loadModel}")
        checkpoint = torch.load(opt.loadModel)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint.cuda()
    else:
        print("No pre-trained model specified via -loadModel. Exiting.")
        return

    # Set model to training mode for overfitting experiments.
    model.eval()

    # Run inference on the selected image.
    # (We use no_grad for visualization, even though we're in train mode.)
    single_input = single_input.cuda()
    with torch.no_grad():
        pred_heatmaps, pred_pafs = model(single_input)

    # If outputs are lists (multi-stage), take the last stage.
    if isinstance(pred_heatmaps, list):
        pred_heatmaps = pred_heatmaps[-1]
    if isinstance(pred_pafs, list):
        pred_pafs = pred_pafs[-1]

    # Remove the batch dimension.
    pred_heatmaps = pred_heatmaps.cpu().squeeze(0)  # [num_hm_channels, H_pred, W_pred]
    pred_pafs     = pred_pafs.cpu().squeeze(0)      # [num_paf_channels, H_pred, W_pred]

    # For visualization, upsample predicted maps to the original image size.
    _, H_orig, W_orig = original_image_tensor.shape
    if pred_heatmaps.shape[1:] != (H_orig, W_orig):
        pred_heatmaps_vis = F.interpolate(pred_heatmaps.unsqueeze(0),
                                          size=(H_orig, W_orig),
                                          mode='bilinear',
                                          align_corners=False).squeeze(0)
    else:
        pred_heatmaps_vis = pred_heatmaps.clone()

    if pred_pafs.shape[1:] != (H_orig, W_orig):
        pred_pafs_vis = F.interpolate(pred_pafs.unsqueeze(0),
                                      size=(H_orig, W_orig),
                                      mode='bilinear',
                                      align_corners=False).squeeze(0)
    else:
        pred_pafs_vis = pred_pafs.clone()

    # Convert predicted maps (for visualization) to NumPy arrays.
    pred_heatmaps_np = pred_heatmaps_vis.numpy()
    pred_pafs_np     = pred_pafs_vis.numpy()
    gt_heatmaps_np   = single_gt_heatmaps.cpu().numpy() if single_gt_heatmaps is not None else None
    gt_pafs_np       = single_gt_pafs.cpu().numpy() if single_gt_pafs is not None else None

    num_hm_channels = pred_heatmaps_np.shape[0]
    num_paf_channels = pred_pafs_np.shape[0]
    print(f"Found {num_hm_channels} heatmap channels and {num_paf_channels} paf channels.")

    # --- Compute Losses (for reference) ---
    if single_gt_heatmaps is not None:
        pred_heatmaps_loss = F.interpolate(pred_heatmaps.unsqueeze(0),
                                           size=single_gt_heatmaps.shape[1:],
                                           mode='bilinear',
                                           align_corners=False)
        loss_hm = criterion_hm(pred_heatmaps_loss.to(torch.float32),
                                single_gt_heatmaps.unsqueeze(0).to(torch.float32)) / (single_gt_heatmaps.shape[0])
        print(f"Heatmap loss: {loss_hm.item()}")
    if single_gt_pafs is not None:
        pred_pafs_loss = F.interpolate(pred_pafs.unsqueeze(0),
                                       size=single_gt_pafs.shape[1:],
                                       mode='bilinear',
                                       align_corners=False)
        loss_paf = criterion_paf(pred_pafs_loss.to(torch.float32),
                                 single_gt_pafs.unsqueeze(0).to(torch.float32)) / (single_gt_pafs.shape[0])
        print(f"PAF loss: {loss_paf.item()}")

    # --- Compute per-pixel error maps (absolute differences) for visualization ---
    if gt_heatmaps_np is not None:
        pred_heatmaps_down = F.interpolate(pred_heatmaps.unsqueeze(0),
                                           size=single_gt_heatmaps.shape[1:],
                                           mode='bilinear',
                                           align_corners=False).squeeze(0).numpy()
        error_map_hm = np.abs(pred_heatmaps_down - gt_heatmaps_np)
    else:
        error_map_hm = None

    if gt_pafs_np is not None:
        pred_pafs_down = F.interpolate(pred_pafs.unsqueeze(0),
                                       size=single_gt_pafs.shape[1:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(0).numpy()
        error_map_paf = np.abs(pred_pafs_down - gt_pafs_np)
    else:
        error_map_paf = None

    # --- Display individual channels ---
    def display_maps(gt_maps, pred_maps, title_prefix, channels_per_fig=4):
        num_channels = pred_maps.shape[0]
        for start in range(0, num_channels, channels_per_fig):
            end = min(start + channels_per_fig, num_channels)
            n = end - start
            fig, axs = plt.subplots(n, 2, figsize=(12, n * 4))
            if n == 1:
                axs = np.expand_dims(axs, axis=0)
            for i, ch in enumerate(range(start, end)):
                print(ch)
                if gt_maps is not None:
                    ax = axs[i, 0]
                    im = ax.imshow(gt_maps[ch], cmap='jet')
                    ax.set_title(f"GT {title_prefix} (Ch {ch})")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax = axs[i, 1]
                im = ax.imshow(pred_maps[ch], cmap='jet')
                ax.set_title(f"Pred {title_prefix} (Ch {ch})")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    # Display heatmaps.
    display_maps(gt_heatmaps_np, pred_heatmaps_np, "Heatmap", channels_per_fig=4)
    # Display PAFs.
    display_maps(gt_pafs_np, pred_pafs_np, "PAF", channels_per_fig=4)

    # --- Create combined overlays ---
    combined_pred_hm = np.sum(pred_heatmaps_np, axis=0)
    combined_pred_hm = (combined_pred_hm - combined_pred_hm.min()) / (combined_pred_hm.max() - combined_pred_hm.min() + 1e-8)
    if gt_heatmaps_np is not None:
        combined_gt_hm = np.sum(gt_heatmaps_np, axis=0)
        combined_gt_hm = (combined_gt_hm - combined_gt_hm.min()) / (combined_gt_hm.max() - combined_gt_hm.min() + 1e-8)
    else:
        combined_gt_hm = None

    original_image = original_image_tensor.cpu().numpy()  # [C, H, W]
    if original_image.ndim == 3 and original_image.shape[0] in [1, 3]:
        original_image = np.transpose(original_image, (1, 2, 0))
    original_image = np.clip(original_image, 0, 1)

    plt.figure(figsize=(12, 10))
    plt.imshow(original_image)
    plt.imshow(combined_pred_hm, cmap='jet', alpha=0.5)
    plt.title("Overlay: Original Image with Combined Predicted Heatmap")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    if combined_gt_hm is not None:
        plt.figure(figsize=(12, 10))
        plt.imshow(original_image)
        plt.imshow(combined_gt_hm, cmap='jet', alpha=0.5)
        plt.title("Overlay: Original Image with Combined GT Heatmap")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    if error_map_hm is not None:
        combined_error_hm = np.sum(error_map_hm, axis=0)
        combined_error_hm = (combined_error_hm - combined_error_hm.min()) / (combined_error_hm.max() - combined_error_hm.min() + 1e-8)
        plt.figure(figsize=(12, 10))
        plt.imshow(original_image)
        plt.imshow(combined_error_hm, cmap='jet', alpha=0.5)
        plt.title("Overlay: Original Image with Heatmap Error")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    if error_map_paf is not None:
        combined_error_paf = np.sum(error_map_paf, axis=0)
        combined_error_paf = (combined_error_paf - combined_error_paf.min()) / (combined_error_paf.max() - combined_error_paf.min() + 1e-8)
        plt.figure(figsize=(12, 10))
        plt.imshow(original_image)
        plt.imshow(combined_error_paf, cmap='jet', alpha=0.5)
        plt.title("Overlay: Original Image with PAF Error")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
