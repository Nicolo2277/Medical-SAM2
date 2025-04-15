'''Data loader for the main dataset for pretraining on binary segmentation'''

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from func_2d.utils import random_click
import torch.nn.functional as F
import matplotlib.pyplot as plt

def invert_mask(background_mask_tensor):
    '''Invert the background binary mask into foregroung mask'''
    return 1.0 - background_mask_tensor

def apply_fan(fan, mask, image):
        '''Apply fan cropping to both masks and images'''
        # Convert fan image to a numpy array and create a binary mask.
        fan_np = np.array(fan)
        threshold = 128  
        binary_fan = fan_np > threshold  #True for pixels inside the fan

        # Get indices where binary_fan is True
        coords = np.column_stack(np.where(binary_fan))
        if coords.size > 0:
            # Compute the bounding box of the fan region.
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Crop both the image and the mask using the bounding box
            crop_box = (x_min, y_min, x_max, y_max)
            image = image.crop(crop_box)
            mask = mask.crop(crop_box)
        return image, mask

class MainDataset(Dataset):
    def __init__(self, args, data_path, transforms_img=None, transform_mask=None, mode='training', prompt='click'):
        self.args = args
        self.data_path = data_path
        self.transforms_img = transforms_img
        self.transform_mask = transform_mask
        self.mode = mode
        self.prompt = prompt
        self.image_size = args.image_size
        self.mask_size = args.out_size

        # List to hold tuples: (frame_path, mask_path)
        self.data = []
        self.fan = []
        # Traverse the folder tree
        # The logic here: walk through all subdirectories. In each folder, check if both
        # 'frame.png' and 'bacground.png' are present. If so, add to the list.
        for root, dirs, files in os.walk(self.data_path):
            if "frame.png" in files and "background.png" in files:
                frame_path = os.path.join(root, "frame.png")
                mask_path = os.path.join(root, "background.png")
                self.data.append((frame_path, mask_path))

                parent_path = os.path.dirname(root)
                fan_path = os.path.join(parent_path, 'fan.png')
                self.fan.append(fan_path)
            

        #print(f"Found {len(self.data)} valid frame-mask pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frame_path, mask_path = self.data[index]
        fan_path = self.fan[index]

        frame = Image.open(frame_path).convert("RGB")
        background_mask = Image.open(mask_path).convert("L")  # Assume mask is grayscale.
        fan = Image.open(fan_path).convert("L")

        #Apply fan cropping:
        frame, background_mask = apply_fan(fan, background_mask, frame)
        
        if self.transforms_img:
            frame = self.transforms_img(frame)
        else:
            frame = transforms.ToTensor()(frame)
        
        if self.transform_mask:
            background_mask= self.transform_mask(background_mask)
        else:
            background_mask = transforms.ToTensor()(background_mask)
        
        mask = invert_mask(background_mask)
        
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

        filename = os.path.basename(frame_path)

        image_meta_dict = {'filename_or_obj':filename}
        return {
            'image': frame,
            'p_label': point_label,
            'pt': pt,
            'mask': mask_resized,
            'mask_ori': mask_ori,
            'image_meta_dict': image_meta_dict,
        }
        


