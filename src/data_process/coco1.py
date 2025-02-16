import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import albumentations as A

from .coco_process_utils import clean_annot, get_ignore_mask, get_heatmap, get_paf, get_keypoints, FLIP_INDICES
from .process_utils import resize, resize_hm_paf, normalize

class CocoDataSet(data.Dataset):
    def __init__(self, data_path, opt, split='train'):
        self.coco_year = 2017
        ann_file = os.path.join(data_path, 'annotations', f'person_keypoints_{split}{self.coco_year}.json')
        self.coco = COCO(ann_file)
        self.split = split
        self.data_path = data_path
        self.do_augment = (split == 'train')
        
        # Load filtered image IDs/annotations
        self.indices = clean_annot(self.coco, data_path, split)
        self.img_dir = os.path.join(data_path, f'{split}{self.coco_year}')
        self.opt = opt
        print(f'Loaded {len(self.indices)} images for {split}')
        
        # In-memory cache for images
        self.image_cache = {}
        
        # Setup Albumentations augmentation pipeline (only if augmenting)
        if self.do_augment:
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=opt.flipAugProb),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=opt.scaleAugFactor,
                        rotate_limit=opt.rotAugFactor * 15,  # adjust as needed
                        p=opt.rotAugProb
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=opt.colorAugFactor,
                        contrast_limit=opt.colorAugFactor,
                        p=0.5
                    )
                ],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
                additional_targets={'mask': 'mask'}
            )
        else:
            self.aug = None

    def load_image(self, img_path):
        # Use cached image if available
        if img_path in self.image_cache:
            return self.image_cache[img_path]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = img.astype(np.float32) / 255.0
        self.image_cache[img_path] = img
        return img

    def get_item_raw(self, index, to_resize=True):
        # Get image ID from indices
        image_id = self.indices[index]
        anno_ids = self.coco.getAnnIds(image_id)
        annots = self.coco.loadAnns(anno_ids)
        
        # Load image info and image
        img_info = self.coco.loadImgs([image_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = self.load_image(img_path)
        
        # Generate ignore mask and keypoints (assumed to be of shape (N, 3): x, y, visibility)
        ignore_mask = get_ignore_mask(self.coco, img, annots)
        keypoints = get_keypoints(self.coco, img, annots)
        
        # If augmenting, convert keypoints to list of (x, y) tuples
        if self.do_augment and self.aug is not None:
            # Convert keypoints (N,3) to a list of (x, y) tuples
            if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 2:
                keypoints_list = [tuple(pt[:2]) for pt in keypoints]
            else:
                keypoints_list = keypoints
            
            # Apply augmentation
            augmented = self.aug(image=img, mask=ignore_mask, keypoints=keypoints_list)
            img = augmented['image']
            ignore_mask = augmented['mask']
            # Reconstruct keypoints as a (N,3) array by adding a default visibility (1) to each keypoint
            aug_keypoints = augmented['keypoints']
            keypoints = np.array([[x, y, 1] for (x, y) in aug_keypoints], dtype=np.float32)
        
        # Resize if necessary
        if to_resize:
            img, ignore_mask, keypoints = resize(img, ignore_mask, keypoints, self.opt.imgSize)
        
        # Generate target heatmaps and PAFs
        heat_map = get_heatmap(self.coco, img, keypoints, self.opt.sigmaHM)
        paf = get_paf(self.coco, img, keypoints, self.opt.sigmaPAF, self.opt.variableWidthPAF)
        
        return img, heat_map, paf, ignore_mask, keypoints

    def __getitem__(self, index):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index)
        img = normalize(img)
        heat_map, paf, ignore_mask = resize_hm_paf(heat_map, paf, ignore_mask, self.opt.hmSize)
        return img, heat_map, paf, ignore_mask, index

    def get_imgs_multiscale(self, index, scales, flip=False):
        img, heat_map, paf, ignore_mask, keypoints = self.get_item_raw(index, to_resize=False)
        imgs = []
        for scale in scales:
            width, height = img.shape[1], img.shape[0]
            new_width = int(scale * width)
            new_height = int(scale * height)
            scaled_img = cv2.resize(img.copy(), (new_width, new_height))
            norm_img = normalize(scaled_img)
            imgs.append(norm_img)
            if flip:
                flip_img = cv2.flip(scaled_img, 1)
                imgs.append(normalize(flip_img))
        paf = paf.transpose(2, 3, 0, 1)
        paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
        paf = paf.transpose(2, 0, 1)
        return imgs, heat_map, paf, ignore_mask, keypoints

    def __len__(self):
        return len(self.indices)
