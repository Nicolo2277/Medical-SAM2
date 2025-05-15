import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from func_2d.utils import random_click
import torch
import torch.nn.functional as F
import random

def cv_random_flip(img, label, p=0.5):
        if random.random() < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label
        
def randomRotation(img, label, p=0.5, angle_range=(-15, 15)):
        if random.random() < p:
            angle = random.uniform(angle_range[0], angle_range[1])
            img = img.rotate(angle, Image.BICUBIC)
            label = label.rotate(angle, Image.NEAREST)
        return img, label
    
def randomCrop(img, label, p=0.3):
        if random.random() < p:
            width, height = img.size
            crop_ratio = random.uniform(0.8, 0.95)
            crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))
            label = label.crop((left, top, left + crop_w, top + crop_h))
            img = img.resize((width, height), Image.BICUBIC)
            label = label.resize((width, height), Image.NEAREST)
        return img, label
    
def colorEnhance(img, intensity='medium'):
        if intensity == 'none':
            return img
        if intensity == 'light':
            bright_range = (0.9, 1.1)
            contrast_range = (0.9, 1.1)
            color_range = (0.9, 1.1)
            sharp_range = (0.9, 1.1)
        elif intensity == 'medium':
            bright_range = (0.7, 1.3)
            contrast_range = (0.7, 1.3)
            color_range = (0.7, 1.3)
            sharp_range = (0.7, 1.3)
        elif intensity == 'heavy':
            bright_range = (0.5, 1.5)
            contrast_range = (0.5, 1.5)
            color_range = (0.5, 1.5)
            sharp_range = (0.5, 1.5)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*bright_range))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast_range))
        img = ImageEnhance.Color(img).enhance(random.uniform(*color_range))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharp_range))
        return img

def randomGamma(img, p=0.3, gamma_range=(0.7, 1.5)):
        if random.random() < p:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            img = Image.fromarray(np.uint8(255 * np.power(np.array(img) / 255.0, gamma)))
        return img
    
    
def randomPeper(img, p=0.3, intensity=0.0015):
        if random.random() < p:
            arr = np.array(img)
            num = int(intensity * arr.size)
            xs = np.random.randint(0, arr.shape[0], num)
            ys = np.random.randint(0, arr.shape[1], num)
            arr[xs, ys] = np.random.choice([0, 255], num)
            return Image.fromarray(arr)
        return img
    
def randomBlur(img, p=0.2):
        if random.random() < p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return img
    
def apply_augmentation(img, mask, augment_intensity='medium'):
        if augment_intensity == 'none':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0, 0, 0, 0, 0, 0
        elif augment_intensity == 'light':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.2, 0.1, 0.1, 0.1, 0.05
        elif augment_intensity == 'medium':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.3, 0.3, 0.2, 0.2, 0.1
        elif augment_intensity == 'heavy':             
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.4, 0.4, 0.3, 0.3, 0.15
        
        img, mask = cv_random_flip(img, mask, p=flip_p)

        img, mask = randomRotation(img, mask, p=rotate_p, angle_range=(-20, 20) if augment_intensity == 'heeavy' else (-15, 15))

        img, mask = randomCrop(img, mask, p = crop_p)

        img = colorEnhance(img, intensity=augment_intensity)
        img = randomBlur(img, p=blur_p)
        img = randomGamma(img, p=gamma_p)
        #img = randomPeper(img, p=peper_p)

        return img, mask

def apply_fan(fan_pil, *pils):
    """
    Crops each PIL in `pils` to the bounding box of `fan_pil`.
    Returns a tuple of cropped PILs in the same order.
    """
    fan_np = np.array(fan_pil)
    mask   = fan_np > 0
    coords = np.column_stack(np.where(mask))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        box = (x0, y0, x1, y1)
        return tuple(p.crop(box) for p in pils)
    else:
        return pils  # no cropping if fan empty


class MainDataset(Dataset):
    """Multiclass version of your original MainDataset with one-hot encoded masks."""
    def __init__(self, args, data_path,
                 transforms_img=None, transform_mask=None,
                 mode='training', prompt='click', max_num=3):
        self.args           = args
        self.data_path      = data_path
        self.transforms_img = transforms_img
        self.transform_mask = transform_mask
        self.mode           = mode
        self.prompt         = prompt
        self.mask_size      = args.out_size
        self.max_num = max_num
        # define number of classes (including background)
        self.num_classes    = 3  # background, solid, non-solid

        self.data = []
        for root, _, files in os.walk(data_path):
            if "frame.png" in files and "background.png" in files:
                frame_fp = os.path.join(root, "frame.png")
                bg_fp    = os.path.join(root, "background.png")
                solid_fp = os.path.join(root, "solid.png")
                non_fp   = os.path.join(root, "non-solid.png")
                parent   = os.path.dirname(root)
                fan_fp   = os.path.join(parent, "fan.png")

                self.data.append((
                    frame_fp,
                    bg_fp,
                    solid_fp if os.path.exists(solid_fp) else None,
                    non_fp   if os.path.exists(non_fp)   else None,
                    fan_fp
                ))
        
        self.samples = []
        if self.max_num is not None and len(self.data) > self.max_num:
                selected_clips = []
                indices = np.linspace(0, len(self.data) - 1, self.max_num, dtype=int)
                for idx in indices:
                    selected_clips.append(self.data[idx])
                self.samples.extend(selected_clips)
        else:
                self.samples.extend(self.data)
        
        print(f"Total data: {len(self.samples)}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_fp, bg_fp, solid_fp, non_fp, fan_fp = self.data[idx]

        # Load images
        frame_pil    = Image.open(frame_fp).convert("RGB")
        bg_pil       = Image.open(bg_fp).convert("L")
        solid_pil    = Image.open(solid_fp).convert("L") if solid_fp else Image.new("L", frame_pil.size, 0)
        nonsolid_pil = Image.open(non_fp).convert("L")   if non_fp   else Image.new("L", frame_pil.size, 0)
        fan_pil      = Image.open(fan_fp).convert("L")

        # Crop all together
        frame_pil, bg_pil, solid_pil, nonsolid_pil = apply_fan(
            fan_pil, frame_pil, bg_pil, solid_pil, nonsolid_pil
        )

        #If no data augmentation except fan cropping then comment those lines:
        frame_pil, bg_pil = apply_augmentation(frame_pil, bg_pil)
        frame_pil, solid_pil = apply_augmentation(frame_pil, solid_pil)
        frame_pil, nonsolid_pil = apply_augmentation(frame_pil, nonsolid_pil)

        # Image transform
        if self.transforms_img:
            frame = self.transforms_img(frame_pil)
        else:
            frame = transforms.ToTensor()(frame_pil)

        # Build label map HxW
        H, W = bg_pil.height, bg_pil.width
        label_arr = np.zeros((H, W), dtype=np.int64)
        solid_mask = np.array(solid_pil) > 0
        label_arr[solid_mask] = 1
        nonsolid_mask = np.array(nonsolid_pil) > 0
        label_arr[nonsolid_mask] = 2

        # Resize label map to mask_size using nearest
        lbl_t = torch.from_numpy(label_arr).long()
        mask_resized = F.interpolate(
            lbl_t.unsqueeze(0).unsqueeze(0).float(),
            size=(self.mask_size, self.mask_size),
            mode='nearest'
        ).long().squeeze(0).squeeze(0)

        # One-hot encode to [num_classes, H, W]
        mask_onehot = F.one_hot(mask_resized, num_classes=self.num_classes)
        # Permute to [C, H, W]
        mask = mask_onehot.permute(2, 0, 1).float()

        # Generate click prompts if needed
        p_labels, pts = [], []
        if self.prompt == 'click':
            for cls in range(1, self.num_classes):
                bm = (mask_resized == cls).float().numpy()
                if bm.sum() > 0:
                    pl, (y, x) = random_click(bm, point_label=cls)
                else:
                    pl, (y, x) = 0, (-1, -1)
                p_labels.append(pl)
                pts.append((y, x))
        else:
            p_labels = [0] * (self.num_classes - 1)
            pts      = [(-1, -1)] * (self.num_classes - 1)

        point_label = torch.tensor(p_labels, dtype=torch.long)
        pt          = torch.tensor(pts,      dtype=torch.long)

        filename = os.path.basename(frame_fp)
        image_meta_dict = {'filename_or_obj': filename}
        #print(mask_resized.shape) (256, 256)
        
        return {
            'image':           frame,
            'p_label':         point_label,
            'pt':              pt,
            'mask':            mask,
            'mask_ori':        mask_resized,
            'image_meta_dict': image_meta_dict,
        }