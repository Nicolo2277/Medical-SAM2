import os
from torchvision import transforms
import argparse
from argparse import Namespace
from func_2d.main_dataset import *
import matplotlib.pyplot as plt


'''
data_path = './Preliminary-data'
i=0
for f in os.scandir(data_path):
    if f.is_dir():
        #print("Folder:", f.name, "Path:", f.path)
        for g in os.scandir(f.path):
            if g.is_dir():
                #print("Subfolder:", g.name, "Path:", g.path)
                name = g.path.split('/')[-1]
                for n in os.scandir(g.path):
                    if n.is_dir():
                        image_count = sum(1 for entry in os.scandir(n.path) if entry.is_file())
                        if image_count > 1:
                            i += 1
print(i)
'''

#to test the dataloader
def test_loader(data_dir):
    # Create a dummy args namespace with necessary attributes.
    args = Namespace(image_size=256, out_size=256)  # Change out_size as needed

    # Define sample transformations for the image and mask.
    transform_img = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor()
    ])

    # Initialize the dataset. Make sure your dataset folder is correctly formatted.
    dataset = MainDataset(args, data_dir, transforms_img=transform_img, transform_mask=transform_mask, prompt='click')
    if len(dataset) == 0:
        print("No valid frame-mask pairs found.")
        return

    print(f"Found {len(dataset)} valid frame-mask pairs.")

    num_samples = 1
    plt.figure(figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]

        # Extract tensors.
        frame = sample['image']
        mask_ori = sample['mask_ori']
        mask_resized = sample['mask']
        meta = sample['image_meta_dict']

        print(f'Sample {i}')
        print('Frame path: ', meta['frame_path'])
        print('Mask path: ', meta['mask_path'])
        print('Fan path: ', meta['fan_path'])

        # Convert tensors to numpy arrays.
        frame_np = frame.permute(1, 2, 0).numpy()
        mask_ori_np = mask_ori.squeeze().numpy()
        mask_resized_np = mask_resized.squeeze().numpy()

        # Create a subplot for this sample with three columns:
        # Original frame, original binary mask, and resized mask.
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.imshow(frame_np)
        ax1.set_title(f"Frame {i}\n{sample['image_meta_dict']['filename_or_obj']}")
        ax1.axis("off")

        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.imshow(mask_ori_np, cmap="gray")
        ax2.set_title(f"mask_ori {i}")
        ax2.axis("off")

        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        ax3.imshow(mask_resized_np, cmap="gray")
        ax3.set_title(f"mask_resized {i}")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = './Preliminary-data'

    test_loader(data_dir)


