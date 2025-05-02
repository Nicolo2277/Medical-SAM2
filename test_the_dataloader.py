import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as Namespace
import torchvision.transforms as T
import torch

from func_2d.multiclass_dataset import MainDataset  

# ----------------------------------------------------------------------------
# Utility: RGB overlay of multiclass mask
# ----------------------------------------------------------------------------
def to_rgb_mask(m):
    cmap = {0:(0,0,0), 1:(1,0,0), 2:(0,0,1)}
    h, w = m.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls, col in cmap.items():
        rgb[m==cls] = col
    return rgb

# ----------------------------------------------------------------------------
# Visualization of dataset samples
# ----------------------------------------------------------------------------
def test_loader(data_dir):
    # 1) dummy args
    args = Namespace(image_size=256, out_size=256)

    # 2) image transforms: resize then tensor
    transform_img = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])

    # 3) init dataset
    dataset = MainDataset(
        args,
        data_dir,
        transforms_img=transform_img,
    )
    n = len(dataset)
    if n == 0:
        print("❌ No valid samples found in:", data_dir)
        return
    print(f"✅ Found {n} samples in {data_dir}\n")

    # 4) iterate and visualize
    for idx in range(min(n, 5)):  # show up to 5 samples
        sample = dataset[idx]
        frame        = sample['image']        # 3×H×W
        mask_ori     = sample['mask_ori'].squeeze(0)     # H×W LongTensor
        mask_resized = sample['mask'].squeeze(0)         # H_out×W_out LongTensor
        p_label      = sample.get('p_label')  # Tensor or None
        pt           = sample.get('pt')       # Tensor([y, x]×N) or None
        meta         = sample['image_meta_dict']

        # 5) class presence
        print(f"Sample {idx}: {meta['filename_or_obj']}")
        print("  - Original mask classes: ", torch.unique(mask_ori).tolist())
        print("  - Resized mask classes:  ", torch.unique(mask_resized).tolist(), "\n")

        # 6) convert to numpy
        frame_np        = frame.permute(1,2,0).cpu().numpy()
        mask_ori_np     = mask_ori.cpu().numpy()
        mask_resized_np = mask_resized.cpu().numpy()

        # 7) plot
        fig, axes = plt.subplots(2, 3, figsize=(15,10))
        fig.suptitle(f"Sample {idx}: {meta['filename_or_obj']}", fontsize=16)

        # A) frame + clicks
        ax = axes[0,0]
        ax.imshow(frame_np)
        ax.set_title("Frame + clicks")
        if pt is not None and p_label is not None:
            pts = pt.cpu().numpy()
            lbls = p_label.cpu().numpy()
            # ensure arrays
            if pts.ndim == 1:
                pts = np.expand_dims(pts, 0)
                lbls = np.expand_dims(lbls, 0)
            h_img, w_img, _ = frame_np.shape
            h_m, w_m = mask_resized_np.shape
            y_scale = h_img / h_m
            x_scale = w_img / w_m
            # iterate points
            for i in range(pts.shape[0]):
                print(pts.shape) #(2, 2)
                y, x = pts[i]
                cls = lbls[i]
                #print('classe: ', cls)
                if y < 0 or x < 0:
                    continue
                color = 'red' if cls==1 else ('blue' if cls==2 else 'yellow')
                ax.plot(
                    x * x_scale,
                    y * y_scale,
                    'o', markeredgecolor='black',
                    markerfacecolor=color,
                    markersize=12,
                    label=f"class {cls}"
                )
            ax.legend(loc='upper right')
        ax.axis("off")

        # B) original multiclass mask
        axes[0,1].imshow(to_rgb_mask(mask_ori_np))
        axes[0,1].set_title("mask_ori (multiclass)")
        axes[0,1].axis("off")

        # C) resized multiclass mask
        axes[0,2].imshow(to_rgb_mask(mask_resized_np))
        axes[0,2].set_title("mask_resized (multiclass)")
        axes[0,2].axis("off")

        # D) solid overlay
        axes[1,0].imshow(frame_np, alpha=0.7)
        axes[1,0].imshow((mask_resized_np==1), cmap="Reds", alpha=0.5)
        axes[1,0].set_title("Solid (class 1)")
        axes[1,0].axis("off")

        # E) nonsolid overlay
        axes[1,1].imshow(frame_np, alpha=0.7)
        axes[1,1].imshow((mask_resized_np==2), cmap="Blues", alpha=0.5)
        axes[1,1].set_title("Non-solid (class 2)")
        axes[1,1].axis("off")

        # F) combined check
        axes[1,2].imshow(to_rgb_mask(mask_resized_np))
        axes[1,2].set_title("Combined check")
        axes[1,2].axis("off")

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.show()

if __name__ == "__main__":
    test_loader("./Preliminary-data")
