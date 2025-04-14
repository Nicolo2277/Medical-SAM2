'''Training script for OTU_2d dataset'''
import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader, random_split

import cfg
import func_2d.function as function
from conf import settings  #maybe global_settings?
#from models.discriminatorlayer import discriminator
from func_2d.OTU_dataset import * #change here if the dataset is changed(or add another dataloader to the import)
from func_2d.utils import *

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

    #optimization
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    '''load the pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    
    #Example on the OTU dataset:
    if args.dataset == 'OTU':
        full_dataset = OTU_2D(args, args.data_path, transform=transform_train)
        #80% training 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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