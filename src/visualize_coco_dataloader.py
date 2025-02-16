from opts.viz_opts import VizOpts
from data_process.coco import CocoDataSet
from visualization.visualize import visualize_masks, visualize_keypoints, visualize_heatmap, visualize_paf
from data_process.coco_process_utils import BODY_PARTS

if __name__ == '__main__':
    opts = VizOpts().parse()
    
    for split in ['train', 'val']:
        print(f"\n[INFO] Processing split: {split}")
        coco_dataset = CocoDataSet(opts.data, opts, split)

        for i in range(len(coco_dataset)):
            img, heatmaps, paf, ignore_mask, keypoints = coco_dataset.get_item_raw(i)
            img = (img * 255.).astype('uint8')

            # Print dimensions
            print(f"\n[INFO] Sample {i+1} from {split} set:")
            print(f"  - Image Shape: {img.shape}")  # Expected: (H, W, C)
            print(f"  - Heatmaps Shape: {heatmaps.shape}")  # Expected: (num_keypoints, H, W)
            print(f"  - PAF Shape: {paf.shape}")  # Expected: (num_pafs, H, W)
            print(f"  - Ignore Mask Shape: {ignore_mask.shape}")  # Expected: (H, W) or (1, H, W)
            print(f"  - Keypoints Length: {len(keypoints)}")  # Number of keypoints
            
            # Visualizations
            visualize_keypoints(img, keypoints, BODY_PARTS)
            if opts.vizIgnoreMask:
                visualize_masks(img, ignore_mask)
            if opts.vizHeatMap:
                visualize_heatmap(img, heatmaps)
            if opts.vizPaf:
                visualize_paf(img, paf)
