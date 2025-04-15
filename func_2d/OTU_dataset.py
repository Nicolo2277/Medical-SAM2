'''Data loader for the OTU_2d dataset'''
import os
import numpy as np
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from func_2d.utils import random_click


class OTU_2D(Dataset):
    def __init__(self, args, data_path , transform = None, transform_mask = None, mode = 'Training',prompt = 'click', plane = False):
        self.images_dir = os.path.join(data_path, 'images')
        self.masks_dir = os.path.join(data_path, 'annotations')
        self.mode = mode
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.[jJ][pP][gG]')))
        self.transform = transform
        self.transform_mask = transform_mask
        self.args = args
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path = self.image_files[index]
        filename = os.path.basename(image_path)
        
        name_without_ext, _ = os.path.splitext(filename)
        mask_filename = name_without_ext + '.PNG'
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        else:
            mask = torch.as_tensor(np.array(mask), dtype=torch.float32)
        
        if self.prompt == 'click':
            mask = mask / mask.max()  # rescale so that the maximum becomes 1
            mask_array = np.array(mask)
            
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
                #print('the mask shape is: ', mask_array.shape)

            point_label, pt = random_click(mask_array, point_label=1)

            mask_ori = (mask >= 0.5).float()  # binarize if needed
            mask_resized = F.interpolate(mask_ori.unsqueeze(0), size=(self.mask_size, self.mask_size),
                                         mode='bilinear', align_corners=False).mean(dim=0)
            mask_resized = (mask_resized >= 0.5).float()

        image_meta_dict = {'filename_or_obj':filename}
        return {
            'image': img,
            'p_label': point_label,
            'pt': pt,
            'mask': mask_resized,
            'mask_ori': mask_ori,
            'image_meta_dict': image_meta_dict,
        }