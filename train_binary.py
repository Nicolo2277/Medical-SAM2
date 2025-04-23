'''Training script for OTU_2d and main dataset'''
import os
import time
import random
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader, random_split, ConcatDataset

import cfg
import func_2d.function as function
from conf import settings  #maybe global_settings?
#from models.discriminatorlayer import discriminator
from func_2d.OTU_dataset import *
from func_2d.utils import *
from func_2d.main_dataset import *

def get_video_dir(root_dir):
    video_dirs = []
    for root, dirs, files in os.walk(root_dir):
         if "frame.png" in files and "background.png" in files:
            video_dirs.append(root)
    return sorted(video_dirs)

def main():
    #Use bfloat16
    torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    '''
    print(">>> Model parameters and trainability:")
    total_params = 0
    trainable_params = 0

    for name, param in net.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

        # cast shape to string so width specifiers work
        shape_str = str(tuple(param.shape))
        print(f"{name:40} | shape: {shape_str:20} | trainable: {param.requires_grad}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(">>> End of parameter list")
    '''

    #optimization
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    OTU_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    main_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    OTU_datapath = './OTU_2d'
    main_datapath = './Preliminary-data'

    OTU_dataset = OTU_2D(args, OTU_datapath, transform=OTU_transforms)
    train_size = int(0.8 * len(OTU_dataset))
    val_size = len(OTU_dataset) - train_size
    train_OTU, val_OTU = random_split(OTU_dataset, [train_size, val_size])

    #Splitting the main dataset:
    all_video_dirs = get_video_dir(main_datapath)
    random.seed(42)
    random.shuffle(all_video_dirs)

    #print(len(all_video_dirs))
    num_train = int(0.8 * len(all_video_dirs))
    train_video_dirs = all_video_dirs[:num_train]
    val_video_dirs = all_video_dirs[num_train:]

    train_main = MainDataset(args, data_path=main_datapath, transforms_img=main_transforms, transform_mask=main_transforms,
                                mode='training', prompt='click')
    train_main.data = [(os.path.join(video_dir, 'frame.png'), os.path.join(video_dir, 'background.png')) 
                          for video_dir in train_video_dirs]
    val_main = MainDataset(args, data_path=main_datapath, transforms_img=main_transforms, transform_mask=main_transforms)
    val_main.data = [(os.path.join(video_dir, 'frame.png'), os.path.join(video_dir, 'background.png')) 
                          for video_dir in val_video_dirs]
    
    #print('The len of the main train dataset is: ', len(train_main))
    #print('The len of the main val dataset is: ', len(val_main))

    train_dataset = ConcatDataset([train_OTU, train_main])
    val_dataset = ConcatDataset([val_OTU, val_main])
   
    print('The len of the full dataset is: ', len(train_dataset) + len(val_dataset))
    print('The len of the train dataset is: ', len(train_dataset))
    print('The len of the val dataset is: ', len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW
    ))
    #create checkpoint folder to save the model:
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    'begin training'
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):
        if epoch == 0:
            tol, (eiou, edice) = function.validation_sam(args, test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IoU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        
        #training
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_loader, epoch, writer)
        logger.info(f'Training loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        #validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice) = function.validation_sam(args, test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if edice > best_dice:
                best_dice = edice
                torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    main()