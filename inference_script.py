import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
from func_2d import utils as fn2dutils
from func_2d.inference import MedicalSam2ImagePredictor
from sam2_train.automatic_mask_generator import SAM2AutomaticMaskGenerator

# from models.discriminatorlayer import discriminator
from func_2d.main_dataset import *

import matplotlib.pyplot as plt
from pathlib import Path

args = cfg.parse_args()

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

GPUdevice = torch.device("cuda", args.gpu_device)

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

#Change here to correctly pass the test set, now passing a single image:
test_img_path = os.path.join('test', 'images.jpg')
test_img = Image.open(test_img_path).convert('RGB')
test_img_transformed = transform_test(test_img)
test_dataset = [test_img_transformed]

test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


iterator = iter(test_loader)

net = fn2dutils.get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
ckpt_path = 'logs/OTU_MedSAM2_2025_04_15_18_35_05/Model/latest_epoch.pth'  #Change in case of changes
checkpoint = torch.load(ckpt_path, map_location=GPUdevice)

net.load_state_dict(checkpoint['model'])

net.eval()

if args.has_prompt :
    predictor = MedicalSam2ImagePredictor(net, args)
    data_loaded = next(iterator)
    predictor.set_image_batch(data_loaded["image"])
    pred_mask, pred, high_res_multimasks = predictor.predict_batch()

else:
    auto_gen = SAM2AutomaticMaskGenerator(
                                        net,
                                        points_per_side=32,
                                        pred_iou_thresh=0.3,
                                        stability_score_thresh=0.0,
                                        crop_n_layers=1
                                        )

    #image_np     = (data_loaded.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    test_img = transforms.Resize((args.image_size, args.image_size))(test_img)
    test_img = np.array(test_img)
    annotations  = auto_gen.generate(test_img)


import matplotlib.pyplot as plt

# Display the image
mask = annotations[0]['segmentation']  # boolean H×W array

plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap='gray')       # display mask in grayscale :contentReference[oaicite:0]{index=0}
plt.axis('off')                     # turn off axes :contentReference[oaicite:1]{index=1}
plt.title("Segmentation Mask")
plt.show()
    