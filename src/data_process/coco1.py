import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import albumentations as A

from .coco_process_utils import clean_annot, get_ignore_mask, get_heatmap, get_paf, get_keypoints, FLIP_INDICES
from .process_utils import resize, resize_hm_paf, normalize

def _prepare_keypoints_for_aug(keypoints):
    """
    Converts keypoints (expected as a numpy array of shape (N,3) or a list)
    into a list of (x, y) tuples.
    """
    if isinstance(keypoints, np.ndarray):
        keypoints = keypoints.tolist()
    return [(float(pt[0]), float(pt[1])) for pt in keypoints]

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
        
        # In-memory image cache
        self.image_cache = {}
        
        # Setup Albumentations augmentation pipeline (for training only)
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
        # Check cache first
        if img_path in self.image_cache:
            return self.image_cache[img_path]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = img.astype(np.float32) / 255.0
        self.image_cache[img_path] = img
        return img

    def get_item_raw(self, index, to_resize=True):
        image_id = self.indices[index]
        anno_ids = self.coco.getAnnIds(image_id)
        annots = self.coco.loadAnns(anno_ids)
        
        img_info = self.coco.loadImgs([image_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = self.load_image(img_path)
        
        ignore_mask = get_ignore_mask(self.coco, img, annots)
        keypoints = get_keypoints(self.coco, img, annots)  # Expected shape: (N,3)
        
        if self.do_augment and self.aug is not None:
            # Convert keypoints to a list of (x,y) tuples using the helper function
            keypoints_list = _prepare_keypoints_for_aug(keypoints)
            augmented = self.aug(image=img, mask=ignore_mask, keypoints=keypoints_list)
            img = augmented['image']
            ignore_mask = augmented['mask']
            aug_keypoints = augmented['keypoints']  # List of (x,y) tuples
            # Rebuild keypoints as (N,3) with default visibility=1
            keypoints = np.array([[x, y, 1] for (x, y) in aug_keypoints], dtype=np.float32)
        
        if to_resize:
            img, ignore_mask, keypoints = resize(img, ignore_mask, keypoints, self.opt.imgSize)
        
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
            new_width, new_height = int(scale * width), int(scale * height)
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
