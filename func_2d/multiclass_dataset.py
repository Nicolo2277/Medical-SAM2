import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from func_2d.utils import random_click
import torch
import torch.nn.functional as F


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
                 mode='training', prompt='click'):
        self.args           = args
        self.data_path      = data_path
        self.transforms_img = transforms_img
        self.transform_mask = transform_mask
        self.mode           = mode
        self.prompt         = prompt
        self.mask_size      = args.out_size
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