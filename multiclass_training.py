'''Training script for multiclass main dataset'''
import os
import time
import random
import torch
import torch.optim as optim
import torchvision.transforms as transforms
#from tensorboardX import SummaryWriter
import wandb
#from dataset import *
from torch.utils.data import DataLoader, random_split, ConcatDataset

import cfg
import func_2d.function_multiclass as function
from conf import settings  #maybe global_settings?
#from models.discriminatorlayer import discriminator
from func_2d.utils import *
from func_2d.multiclass_dataset import MainDataset

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

    for fold in range(args.num_folds):
        # initialize wandb run
        wandb.init(
            project="MedSAM-2 final",
            name=f'multiclass Mask Segmentation fold {fold}',
            config=dict(vars(args))  # logs all your args as hyperparameters
            )
        
        random.seed(args.random_seed)
        print('Starting training for fold number ', fold)

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

        main_transforms = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])

        input_root = 'Multiclass_Folds/fold_' + str(fold)

        train_path = os.path.join(input_root, 'train')
        full_train_dataset = MainDataset(args=args, data_path=train_path, transforms_img=main_transforms)

        val_path = os.path.join(input_root, 'val')
        full_val_dataset = MainDataset(args=args, data_path=val_path, transforms_img=main_transforms)               

        
        #print('The len of the main train dataset is: ', len(train_main))
        #print('The len of the main val dataset is: ', len(val_main))

    
        print('The len of the full dataset is: ', len(full_train_dataset) + len(full_val_dataset))
        print('The len of the train dataset is: ', len(full_train_dataset))
        print('The len of the val dataset is: ', len(full_val_dataset))
        
        train_loader = DataLoader(full_train_dataset, batch_size=args.b, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(full_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        '''checkpoint path'''
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
        '''
        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW
        ))
        '''
        #create checkpoint folder to save the model:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        'begin training'
        best_tol = 1e4
        best_dice = 0.0

        for epoch in range(settings.EPOCH):
            if epoch == 0:
                val_loss, mean_dice = function.validation_sam(args, test_loader, epoch, net)
                logger.info(f'Val loss: {val_loss}, DICE: {mean_dice} || @ epoch {epoch}.')
            
            #training
            net.train()
            time_start = time.time()
            loss = function.train_sam(args, net, optimizer, train_loader, epoch)
            logger.info(f'Training loss: {loss} || @ epoch {epoch}.')
            time_end = time.time()
            print('time_for_training ', time_end - time_start)

            #validation
            net.eval()
            if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
                val_loss, mean_dice = function.validation_sam(args, test_loader, epoch, net)
                logger.info(f'Val loss: {val_loss}, DICE: {mean_dice} || @ epoch {epoch}.')

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


        wandb.finish()


if __name__ == '__main__':
    main()

    '''
    Change memory encoder by putting 3, 3 instead of 1, 1 in mask input and output
    Change in build sam how the weights for the decoder are initialized 
    '''