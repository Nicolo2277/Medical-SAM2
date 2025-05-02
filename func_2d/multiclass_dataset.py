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
    """Multiclass version of your original MainDataset."""
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

        self.data = []  # list of (frame_fp, bg_fp, solid_fp_or_None, nonsolid_fp_or_None, fan_fp)
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

        # 1) Load all PILs
        frame_pil    = Image.open(frame_fp).convert("RGB")
        bg_pil       = Image.open(bg_fp).convert("L")
        solid_pil    = Image.open(solid_fp).convert("L") if solid_fp else Image.new("L", frame_pil.size, 0)
        nonsolid_pil = Image.open(non_fp).convert("L")   if non_fp   else Image.new("L", frame_pil.size, 0)
        fan_pil      = Image.open(fan_fp).convert("L")

        # 2) Crop *all* of them together
        frame_pil, bg_pil, solid_pil, nonsolid_pil = apply_fan(
            fan_pil, frame_pil, bg_pil, solid_pil, nonsolid_pil
        )

        # 3) To image tensor
        if self.transforms_img:
            frame = self.transforms_img(frame_pil)
        else:
            frame = transforms.ToTensor()(frame_pil)

        # 4) Build the label array (H×W) after cropping
        H, W = bg_pil.height, bg_pil.width
        label_arr = np.zeros((H, W), dtype=np.int64)  # default background=0

        # solid → class 1
        solid_mask = np.array(solid_pil) > 0
        label_arr[solid_mask] = 1

        # non_solid → class 2
        nonsolid_mask = np.array(nonsolid_pil) > 0
        label_arr[nonsolid_mask] = 2

        # 5) Convert to tensors & resize with nearest
        lbl_t = torch.from_numpy(label_arr)  # Long H×W

        mask_ori = F.interpolate(
            lbl_t.unsqueeze(0).unsqueeze(0).float(),
            size=(self.mask_size, self.mask_size),
            mode='nearest'
        ).long().squeeze(0).squeeze(0)

        mask_resized = mask_ori.clone()

        # 6) Click prompts per class → Tensors so .to(device) works
        # We keep order [class1, class2]
        p_labels = []
        pts      = []
        if self.prompt == 'click':
            for cls in (1, 2):
                bm = (mask_resized == cls).float().numpy()
                if bm.sum() > 0:
                    pl, (y, x) = random_click(bm, point_label=cls)
                else:
                    pl, (y, x) = 0, (-1, -1)
                p_labels.append(pl)
                pts.append((y, x))
        else:
            # if no prompt, fill zeros / -1
            p_labels = [0, 0]
            pts      = [(-1, -1), (-1, -1)]

        # convert to tensors
        # p_label: LongTensor [2] with point_label for classes 1 and 2
        point_label = torch.tensor(p_labels, dtype=torch.long)
        # pt: LongTensor [2,2] with coordinates [[y1,x1],[y2,x2]]
        pt          = torch.tensor(pts,      dtype=torch.long)

        #print('p labels: ', p_labels) #[1, 2]
        #print('pts: ', pts) #(2, 2)

        # 7) Meta
        filename = os.path.basename(frame_fp)
        image_meta_dict = {'filename_or_obj': filename}
        #print(mask_resized.shape) #torch.Size([1, 256, 256])

        return {
            'image':           frame,
            'p_label':         point_label,
            'pt':              pt,
            'mask':            mask_resized,
            'mask_ori':        mask_ori,
            'image_meta_dict': image_meta_dict,
        }
