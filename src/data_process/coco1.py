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
        ann_file = os.path.join(data_path, f'annotations/person_keypoints_{split}{self.coco_year}.json')
        self.coco = COCO(ann_file)
        self.split = split
        self.data_path = data_path
        self.do_augment = (split == 'train')
        
        # load annotations that meet specific standards
        self.indices = clean_annot(self.coco, data_path, split)
        self.img_dir = os.path.join(data_path, split + str(self.coco_year))
        self.opt = opt
        print('Loaded {} images for {}'.format(len(self.indices), split))
        
        # In-memory cache for images
        self.image_cache = {}

        # Setup Albumentations augmentation pipeline
        if self.do_augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=opt.flipAugProb),  # Flip with probability from opts
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
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               additional_targets={'mask': 'mask'})
        else:
            self.aug = None

    def get_item_raw(self, index, to_resize=True):
        index = self.indices[index]
        anno_ids = self.coco.getAnnIds(index)
        annots = self.coco.loadAnns(anno_ids)
        
        img_info = self.coco.loadImgs([index])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = self.load_image(img_path)
        
        ignore_mask = get_ignore_mask(self.coco, img, annots)
        keypoints = get_keypoints(self.coco, img, annots)  # likely returns shape (N, 3)
        
        if self.do_augment and self.aug is not None:
            # Convert keypoints to list of 2-tuples (discard the 3rd value if present)
            if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] > 2:
                keypoints_list = [tuple(pt[:2]) for pt in keypoints]
            else:
                keypoints_list = keypoints  # if already in the right format
            
            augmented = self.aug(image=img, mask=ignore_mask, keypoints=keypoints_list)
            img = augmented['image']
            ignore_mask = augmented['mask']
            keypoints = np.array(augmented['keypoints'])  # now in shape (N, 2)
        
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

    def load_image(self, img_path):
        # Use cache to avoid disk I/O every time
        if img_path in self.image_cache:
            return self.image_cache[img_path]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = img.astype('float32') / 255.
        self.image_cache[img_path] = img
        return img

    def get_imgs_multiscale(self, index, scales, flip=False):
        img, heat_map, paf, ignore_mask, keypoints = self.get_item_raw(index, to_resize=False)
        imgs = []
        for scale in scales:
            width, height = img.shape[1], img.shape[0]
            new_width, new_height = int(scale * width), int(scale * height)
            scaled_img = cv2.resize(img.copy(), (new_width, new_height))
            normalized_img = normalize(scaled_img)
            imgs.append(normalized_img)
            if flip:
                flip_img = cv2.flip(scaled_img, 1)
                imgs.append(normalize(flip_img))
        paf = paf.transpose(2, 3, 0, 1)
        paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
        paf = paf.transpose(2, 0, 1)
        return imgs, heat_map, paf, ignore_mask, keypoints

    def __len__(self):
        return len(self.indices)
