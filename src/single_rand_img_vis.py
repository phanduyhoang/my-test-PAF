import argparse
import os
import torch
import cv2
import numpy as np
from data_process.process_utils import resize_hm
from visualization.visualize import visualize_heatmap, visualize_paf

# **1. Parse Arguments from CLI**
parser = argparse.ArgumentParser()
parser.add_argument("-data", required=True, help="Path to dataset or image folder")
parser.add_argument("-loadModel", required=True, help="Path to the trained model")
parser.add_argument("-img", required=False, default="", help="Path to a single image (optional)")
args = parser.parse_args()

# **2. Validate Inputs**
if not os.path.exists(args.loadModel):
    raise FileNotFoundError(f"Model file not found: {args.loadModel}")
if args.img and not os.path.exists(args.img):
    raise FileNotFoundError(f"Image file not found: {args.img}")

# **3. Load Model**
print(f"Loading model from: {args.loadModel}")
model = torch.load(args.loadModel, map_location="cuda")
model = model.eval().cuda()
print("Model loaded!")

# **4. Load & Preprocess Image**
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Image at {image_path} not found!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (368, 368))  # Resize for model input
    image_input = image_resized.astype(np.float32) / 255.0  # Normalize
    image_input = image_input.transpose(2, 0, 1)  # Convert to (C, H, W) format
    image_tensor = torch.tensor(image_input).unsqueeze(0).cuda()
    
    return image_tensor, image_resized

if args.img:
    print(f"Processing image: {args.img}")
    image_tensor, original_image = preprocess_image(args.img)

    # **5. Run Inference**
# Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)  # Get output from model

    # **Fix: Ensure the output is a tuple or list**
    if isinstance(output, (list, tuple)):
        # **Print output type and length for debugging**
        print(f"Model returned type: {type(output)}, length: {len(output)}")

        # **Check if outputs are tensors**
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"Output {i} is a tensor with shape: {out.shape}")
            else:
                print(f"Output {i} is NOT a tensor. Found: {type(out)}")

        # **Extract PAF & Heatmap (modify based on debug info)**
        paf, heatmap = output[-2], output[-1]  # Modify index if needed
    else:
        raise ValueError(f"Unexpected model output format: {type(output)}. Expected list or tuple.")

    # **Ensure PAF & Heatmap are tensors before calling .cpu()**
    # **Ensure PAF & Heatmap are Tensors**
    if isinstance(paf, list):
        print(f"Converting PAF from list to tensor: Length {len(paf)}")
        paf = torch.tensor(np.array(paf)).cuda()
    if isinstance(heatmap, list):
        print(f"Converting Heatmap from list to tensor: Length {len(heatmap)}")
        heatmap = torch.tensor(np.array(heatmap)).cuda()

    # Convert to NumPy
    paf, heatmap = paf.cpu().numpy(), heatmap.cpu().numpy()




    print("Inference complete!")

    # **6. Resize Outputs to Match Image Size**
    heatmap_resized = resize_hm(heatmap[0], (original_image.shape[1], original_image.shape[0]))
    paf_resized = resize_hm(paf[0], (original_image.shape[1], original_image.shape[0]))

    # **7. Visualize Outputs**
    print("Visualizing heatmap and PAFs...")
    visualize_heatmap(original_image, heatmap_resized, "Predicted Heatmap")
    visualize_paf(original_image, paf_resized, "Predicted PAF")

    # Save the output images
    # cv2.imwrite(f"/kaggle/working/heatmap_output.jpg", heatmap_resized)
    # cv2.imwrite(f"/kaggle/working/paf_output.jpg", paf_resized)
    # print("Saved visualization images to /kaggle/working/")
