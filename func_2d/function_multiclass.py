import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#from torchmetrics.classification import MulticlassConfusionMatrix
import cfg
from conf import settings
from func_2d.multiclass_utils import *
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import func_2d.misc2 as misc2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



args = cfg.parse_args()



GPUdevice = torch.device('cuda', args.gpu_device)


pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.CrossEntropyLoss()
mask_type = torch.float32

torch.backends.cudnn.benchmark = True

def class_balanced_focal_loss(logits, targets, num_classes, gamma=2.0, alpha=None):
    """
    Class-balanced focal loss
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,…,C-1}
        num_classes: Number of classes
        gamma: Focusing parameter
        alpha: Optional class weights
        
    Returns: scalar loss
    """
    N, C, H, W = logits.shape

    #print(logits.shape) #[16, 3, 256, 256]
    #print(targets.shape) #[16, 256, 256]
    
    
    # If no specific class weights are provided, calculate them based on inverse frequency
    if alpha is None:
        # Count class frequencies
        class_counts = []
        for c in range(num_classes):
            class_counts.append((targets == c).sum().float() + 1e-6)  # Add small eps to avoid div by 0
        
        # Inverse frequency as weight
        total_pixels = N * H * W
        class_weights = total_pixels / (num_classes * torch.tensor(class_counts).to(logits.device))
        
        # Normalize weights to sum to 1
        alpha = class_weights / class_weights.sum()
    else:
        alpha = torch.tensor(alpha).to(logits.device)
    
    # Compute focal loss with class balancing
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    focal_losses = []
    for c in range(C):
        # Extract class probability and target
        p_c = probs[:, c, ...]
        t_c = targets_onehot[:, c, ...]
        
        # Focal weight: (1-p)^gamma for positive examples, p^gamma for negative examples
        focal_weight = t_c * (1 - p_c)**gamma + (1 - t_c) * p_c**gamma
        
        # Binary cross entropy with focus weight
        bce = -t_c * torch.log(p_c + 1e-6) - (1 - t_c) * torch.log(1 - p_c + 1e-6)
        
        # Apply focal weight and class balancing weight
        class_loss = (alpha[c] * focal_weight * bce).mean()
        focal_losses.append(class_loss)
    
    return sum(focal_losses)


def train_sam(args, net: nn.Module, optimizer, train_loader, epoch):
    
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    
    # train mode
    net.train()
    optimizer.zero_grad()

    # init
    epoch_loss = 0
    memory_bank_list = []
    lossfunc1 = criterion_G
    feat_sizes = [(64, 64), (32, 32), (16, 16)] #If you change the image dimensions[(256, 256), (128, 128), (64, 64)]


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            # input image and gt masks
            imgs = pack['image'].to(dtype = mask_type, device = GPUdevice)
            masks = pack['mask'].to(dtype = mask_type, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            # click prompt: unsqueeze to indicate only one click, add more click across this dimension
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)          # shape: (B, 2) or (B, N, 2)
                pl_temp = pack['p_label'].to(device=GPUdevice)     # shape: (B,) or (B, N)

                # ── Only unsqueeze binary case to make it (B, 1, 2)/(B, 1) ──
                if pt_temp.ndim == 2:
                    # Binary segmentation: one click per image
                    pt = pt_temp.unsqueeze(1)          # now (B, 1, 2)
                    point_labels = pl_temp.unsqueeze(1) # now (B, 1)
                else:
                    # Multiclass: already one click per class, shape (B, N, 2)/(B, N)
                    pt = pt_temp                        # keep (B, N, 2)
                    point_labels = pl_temp             # keep (B, N)

                coords_torch = pt.to(dtype=torch.float, device=GPUdevice)   # (B, M, 2)
                labels_torch = point_labels.to(dtype=torch.int, device=GPUdevice)  # (B, M)
            else:
                coords_torch = None
                labels_torch = None


            '''Train image encoder'''                    
            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            # dimension hint for your future use
            # vision_feats: list: length = 3
            # vision_feats[0]: torch.Size([65536, batch, 32])
            # vision_feats[1]: torch.Size([16384, batch, 64])
            # vision_feats[2]: torch.Size([4096, batch, 256])
            # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
            # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
            # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
            
            

            '''Train memory attention to condition on meomory bank'''         
            B = vision_feats[-1].size(1)  # batch size 
            
            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                
            else:
                for element in memory_bank_list:
                    to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
                    to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
                    to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
 
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64) 
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                
                similarity_scores = F.softmax(similarity_scores, dim=1) 
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )


            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed


            '''prompt encoder'''         
            with torch.no_grad():
                if (ind%5) == 0:
                    points=(coords_torch, labels_torch) # input shape: ((batch, n, 2), (batch, n))
                    flag = True
                else:
                    points=None
                    flag = False

                se, de = net.sam_prompt_encoder(
                    points=points, #(coords_torch, labels_torch)
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])



            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=True, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            
            #print(masks.shape)  #torch.Size([4, 1, 256, 256])          

            

            '''memory encoder'''       
            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
                
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             image_embed[batch].reshape(-1).detach()])
            
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     image_embed[batch].reshape(-1).detach()])

            # backpropagation
            #print(pred.shape)torch.Size([4, 1, 256, 256])
            #print(masks.shape) #torch.Size([4, 256, 256])
            
            targets = masks.argmax(dim=1)

            

            loss = (0.5 * lossfunc1(pred, masks)) + (0.5 * class_balanced_focal_loss(pred, targets, num_classes=3))
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()

            wandb.log({
            "train/loss_batch": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            # can add also memory_bank_size = len(memory_bank_list)
            }, step=epoch * len(train_loader) + ind)
            
            optimizer.step()
            
            optimizer.zero_grad()

            pbar.update()

    avg_loss = epoch_loss / len(train_loader)
    wandb.log({"train/loss_epoch": avg_loss})

    return epoch_loss/len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    """
    Validation function with enhanced metrics tracking for multiclass segmentation.
    Incorporates comprehensive metrics from Vivim model.
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # eval mode
    net.eval()

    n_val = len(val_loader) 
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    # init
    lossfunc1 = criterion_G
    memory_bank_list = []
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    total_loss = 0.0
    C = args.num_classes  # Number of classes

    # Initialize MulticlassMetricsTracker
    multiclass_metrics = MulticlassMetricsTracker(num_classes=C)
    
    # Initialize validation losses and metrics lists
    val_losses = []
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    # For visualization samples
    vis_samples = {
        'images': [],
        'preds': [],
        'targets': []
    }
    num_vis_samples = 10
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pl_temp = pack['p_label'].to(device=GPUdevice)

                # Only unsqueeze binary case to make it (B, 1, 2)/(B, 1)
                if pt_temp.ndim == 2:
                    pt = pt_temp.unsqueeze(1)
                    point_labels = pl_temp.unsqueeze(1)
                else:
                    pt = pt_temp
                    point_labels = pl_temp

                coords_torch = pt.to(dtype=torch.float, device=GPUdevice)
                labels_torch = point_labels.to(dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            with torch.no_grad():
                # image encoder
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1) 

                # memory condition
                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True))
                        
                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64) 
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                    similarity_scores = F.softmax(similarity_scores, dim=1) 
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                # prompt encoder
                if (ind % 5) == 0:
                    flag = True
                    points = (coords_torch, labels_torch)
                else:
                    flag = False
                    points = None

                se, de = net.sam_prompt_encoder(
                    points=points, 
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=True, 
                    repeat_image=False,  
                    high_res_features=high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            
                # memory encoder
                maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag
                )
                    
                maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

                # memory bank
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0],
                                                 image_embed[batch].reshape(-1).detach()])
                else:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores) 
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index) 
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0],
                                                         image_embed[batch].reshape(-1).detach()])

                # Calculate batch loss
                                
               
                targets = masks.argmax(dim=1)

            
                batch_loss = (0.5 * lossfunc1(pred, masks)) + (0.5 * class_balanced_focal_loss(pred, targets, num_classes=3))

                total_loss += batch_loss
                val_losses.append(batch_loss)
                
                # Convert predictions to class indices for confusion matrix
                pred_classes = pred.detach().cpu().argmax(dim=1).numpy().flatten()
                target_classes = masks.detach().cpu().argmax(dim=1).numpy().flatten()
                
                # Store for confusion matrix
                all_preds.extend(pred_classes)
                all_targets.extend(target_classes)
                
                # Update the multiclass metrics tracker
                # Convert tensors to numpy for the tracker
                pred_np = pred.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                multiclass_metrics.update(pred_np, masks_np)
                
                # Collect visualization samples (up to num_vis_samples)
                if len(vis_samples['images']) < num_vis_samples:
                    # Take the first image in the batch
                    vis_samples['images'].append(imgs[0].detach().cpu())
                    vis_samples['preds'].append(pred[0].detach().cpu())
                    vis_samples['targets'].append(masks[0].detach().cpu())
                
                # Update progress bar
                pbar.update(1)
             
    # Calculate average loss
    avg_loss = total_loss / n_val
    
    # Get results from the multiclass metrics tracker
    results = multiclass_metrics.get_results()
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(C))

    # Create figures for confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    class_names = ["background", "solid", "non-solid"]

    # Raw counts confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title(f'Raw Confusion Matrix - Epoch {epoch}')

    # Row-normalized confusion matrix (shows recall)
    row_sums = conf_matrix.sum(axis=1)
    row_norm_conf = conf_matrix / row_sums[:, np.newaxis]
    sns.heatmap(row_norm_conf, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title(f'Row-Normalized Confusion Matrix (Recall) - Epoch {epoch}')

    # Column-normalized confusion matrix (shows precision)
    col_sums = conf_matrix.sum(axis=0)
    col_norm_conf = conf_matrix / col_sums[np.newaxis, :]
    sns.heatmap(col_norm_conf, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[2])
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_title(f'Column-Normalized Confusion Matrix (Precision) - Epoch {epoch}')

    plt.tight_layout()

    # Log confusion matrices to wandb
    wandb.log({
        'val/confusion_matrix_raw': wandb.Image(axes[0].figure),
        'val/confusion_matrix_row_norm': wandb.Image(axes[1].figure),
        'val/confusion_matrix_col_norm': wandb.Image(axes[2].figure),
        'val/loss': avg_loss,
        'Dice': results['dice']['mean'],
        'Jaccard': results['jaccard']['mean'],
        'Precision': results['precision']['mean'],
        'Recall': results['recall']['mean'],
        'Fmeasure': results['f_measure']['mean'],
        'Specificity': results['specificity']['mean']
    })

    # Log per-class metrics
    for i in range(C):
        # Only log if this class appeared during validation
        if results['class_counts'][i] > 0:
            wandb.log({
                f"Dice_class_{class_names[i]}": results['dice']['per_class'][i],
                f"Jaccard_class_{class_names[i]}": results['jaccard']['per_class'][i],
                f"Precision_class_{class_names[i]}": results['precision']['per_class'][i],
                f"Recall_class_{class_names[i]}": results['recall']['per_class'][i],
                f"Fmeasure_class_{class_names[i]}": results['f_measure']['per_class'][i],
                f"Specificity_class_{class_names[i]}": results['specificity']['per_class'][i]
            })

    # Log sample visualizations and the rest of the code remains the same...
        
        # Log per-class metrics
        for i in range(C):
            # Only log if this class appeared during validation
            if results['class_counts'][i] > 0:
                wandb.log({
                    f"Dice_class_{class_names[i]}": results['dice']['per_class'][i],
                    f"Jaccard_class_{class_names[i]}": results['jaccard']['per_class'][i],
                    f"Precision_class_{class_names[i]}": results['precision']['per_class'][i],
                    f"Recall_class_{class_names[i]}": results['recall']['per_class'][i],
                    f"Fmeasure_class_{class_names[i]}": results['f_measure']['per_class'][i],
                    f"Specificity_class_{class_names[i]}": results['specificity']['per_class'][i]
                })
    
    # Log sample visualizations
    if len(vis_samples['images']) > 0:
        # Visualization helper function
        def visualize_samples(images, preds, targets, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            """Create grid of images with predictions and ground truth"""
            num_samples = len(images)
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
                
            class_colors = [
                [0, 0, 0],       # background - black
                [255, 0, 0],     # solid - red
                [0, 0, 255]      # non-solid - blue
            ]
            
            for i in range(num_samples):
                # Original image
                img = images[i].numpy()
                for c in range(3):
                    img[c] = img[c] * std[c] + mean[c]
                img = np.clip(img.transpose(1, 2, 0), 0, 1)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title("Original Image")
                axes[i, 0].axis('off')
                
                # Ground truth
                mask_vis = np.zeros((targets[i].shape[1], targets[i].shape[2], 3))
                target_cls = targets[i].argmax(dim=0).numpy()
                for c in range(C):
                    mask_vis[target_cls == c] = np.array(class_colors[c]) / 255.0
                axes[i, 1].imshow(mask_vis)
                axes[i, 1].set_title("Ground Truth")
                axes[i, 1].axis('off')
                
                # Prediction
                pred_vis = np.zeros((preds[i].shape[1], preds[i].shape[2], 3))
                pred_cls = preds[i].argmax(dim=0).numpy()
                for c in range(C):
                    pred_vis[pred_cls == c] = np.array(class_colors[c]) / 255.0
                axes[i, 2].imshow(pred_vis)
                axes[i, 2].set_title("Prediction")
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            return fig
        
        # Create visualization figure
        vis_fig = visualize_samples(
            vis_samples['images'], 
            vis_samples['preds'], 
            vis_samples['targets']
        )
        
        # Log to wandb
        wandb.log({'val/sample_predictions': wandb.Image(vis_fig)})
        plt.close(vis_fig)
    
    # Report class counts
    print(f"Class counts during validation: {results['class_counts']}")
    
    # Print summary
    print(f"Val: Dice {results['dice']['mean']:.4f}, "
          f"Jaccard {results['jaccard']['mean']:.4f}, "
          f"Precision {results['precision']['mean']:.4f}, "
          f"Recall {results['recall']['mean']:.4f}, "
          f"Fmeasure {results['f_measure']['mean']:.4f}, "
          f"Specificity {results['specificity']['mean']:.4f}")
    
    # Per-class metrics display
    for i, name in enumerate(class_names):
        if results['class_counts'][i] > 0:
            print(f"  Class {name}: Dice {results['dice']['per_class'][i]:.4f}, "
                  f"Jaccard {results['jaccard']['per_class'][i]:.4f}")
    
    # Return average loss and dice score for checkpoint management
    return (
        avg_loss,
        results['dice']['mean']
    )

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F


class MulticlassMetricsTracker:
    """
    A class for tracking and computing various evaluation metrics for multiclass segmentation tasks.
    
    This class calculates the following metrics for each class and as mean values:
    - Dice coefficient (F1-score)
    - Jaccard index (IoU)
    - Precision
    - Recall
    - F-measure
    - Specificity
    
    It handles multiclass prediction data with one-hot encoded masks.
    """
    
    def __init__(self, num_classes: int, eps: float = 1e-8):
        """
        Initialize the MulticlassMetricsTracker.
        
        Args:
            num_classes: Number of classes in the segmentation task
            eps: Small epsilon value to avoid division by zero
        """
        self.num_classes = num_classes
        self.eps = eps
        self.reset()
        
    def reset(self):
        """Reset all accumulated statistics."""
        # Initialize statistics counters for each class
        self.tp = np.zeros(self.num_classes)  # True positives
        self.fp = np.zeros(self.num_classes)  # False positives
        self.tn = np.zeros(self.num_classes)  # True negatives
        self.fn = np.zeros(self.num_classes)  # False negatives
        
        # Track class counts for reporting
        self.class_counts = np.zeros(self.num_classes)
        
        # For direct dice and jaccard computation
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.pred_sum = np.zeros(self.num_classes)
        self.target_sum = np.zeros(self.num_classes)
    
    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        Update metrics with a new batch of predictions and targets.
        
        Args:
            preds: Prediction tensor with shape (B, C, H, W) or softmax probabilities
            targets: Target tensor with shape (B, C, H, W) as one-hot encoded masks
        """
        # Convert to numpy if tensors
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # Get predictions as class indices if not already
        if preds.shape[1] == self.num_classes:  # One-hot or softmax format
            pred_indices = np.argmax(preds, axis=1)  # (B, H, W)
        else:
            pred_indices = preds
            
        # Get target class indices if not already
        if targets.shape[1] == self.num_classes:  # One-hot format
            target_indices = np.argmax(targets, axis=1)  # (B, H, W)
        else:
            target_indices = targets
            
        # Count class occurrences in targets
        for c in range(self.num_classes):
            self.class_counts[c] += np.sum(target_indices == c)
        
        # Calculate metrics for each class
        batch_size = pred_indices.shape[0]
        
        for c in range(self.num_classes):
            for b in range(batch_size):
                # Create binary masks for current class
                pred_mask = (pred_indices[b] == c)
                target_mask = (target_indices[b] == c)
                
                # Calculate confusion matrix elements
                self.tp[c] += np.sum(pred_mask & target_mask)
                self.fp[c] += np.sum(pred_mask & ~target_mask)
                self.fn[c] += np.sum(~pred_mask & target_mask)
                self.tn[c] += np.sum(~pred_mask & ~target_mask)
                
                # Direct calculations for Dice and Jaccard
                intersection = np.sum(pred_mask & target_mask)
                self.intersection[c] += intersection
                self.union[c] += np.sum(pred_mask | target_mask)
                self.pred_sum[c] += np.sum(pred_mask)
                self.target_sum[c] += np.sum(target_mask)
    
    def compute_dice(self) -> Tuple[np.ndarray, float]:
        """
        Compute the Dice coefficient for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class Dice coefficients
            - Mean Dice coefficient across classes with instances
        """
        dice_per_class = 2 * self.intersection / (self.pred_sum + self.target_sum + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_dice = np.mean(dice_per_class[valid_classes])
        else:
            mean_dice = 0.0
            
        return dice_per_class, mean_dice
    
    def compute_jaccard(self) -> Tuple[np.ndarray, float]:
        """
        Compute the Jaccard index (IoU) for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class Jaccard indices
            - Mean Jaccard index across classes with instances
        """
        jaccard_per_class = self.intersection / (self.union + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_jaccard = np.mean(jaccard_per_class[valid_classes])
        else:
            mean_jaccard = 0.0
            
        return jaccard_per_class, mean_jaccard
    
    def compute_precision(self) -> Tuple[np.ndarray, float]:
        """
        Compute precision for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class precision values
            - Mean precision across classes with instances
        """
        precision_per_class = self.tp / (self.tp + self.fp + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_precision = np.mean(precision_per_class[valid_classes])
        else:
            mean_precision = 0.0
            
        return precision_per_class, mean_precision
    
    def compute_recall(self) -> Tuple[np.ndarray, float]:
        """
        Compute recall for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class recall values
            - Mean recall across classes with instances
        """
        recall_per_class = self.tp / (self.tp + self.fn + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_recall = np.mean(recall_per_class[valid_classes])
        else:
            mean_recall = 0.0
            
        return recall_per_class, mean_recall
    
    def compute_f_measure(self) -> Tuple[np.ndarray, float]:
        """
        Compute F-measure (F1 score) for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class F-measure values
            - Mean F-measure across classes with instances
        """
        precision_per_class, _ = self.compute_precision()
        recall_per_class, _ = self.compute_recall()
        
        f_measure_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_f_measure = np.mean(f_measure_per_class[valid_classes])
        else:
            mean_f_measure = 0.0
            
        return f_measure_per_class, mean_f_measure
    
    def compute_specificity(self) -> Tuple[np.ndarray, float]:
        """
        Compute specificity for each class and the mean.
        
        Returns:
            Tuple containing:
            - Array of per-class specificity values
            - Mean specificity across classes with instances
        """
        specificity_per_class = self.tn / (self.tn + self.fp + self.eps)
        
        # Only consider classes that appeared in the dataset
        valid_classes = self.class_counts > 0
        if np.sum(valid_classes) > 0:
            mean_specificity = np.mean(specificity_per_class[valid_classes])
        else:
            mean_specificity = 0.0
            
        return specificity_per_class, mean_specificity
    
    def get_results(self) -> Dict:
        """
        Get a dictionary with all computed metrics.
        
        Returns:
            Dictionary containing all metrics (per-class and mean values)
        """
        dice_per_class, mean_dice = self.compute_dice()
        jaccard_per_class, mean_jaccard = self.compute_jaccard()
        precision_per_class, mean_precision = self.compute_precision()
        recall_per_class, mean_recall = self.compute_recall()
        f_measure_per_class, mean_f_measure = self.compute_f_measure()
        specificity_per_class, mean_specificity = self.compute_specificity()
        
        return {
            'dice': {
                'per_class': dice_per_class,
                'mean': mean_dice
            },
            'jaccard': {
                'per_class': jaccard_per_class,
                'mean': mean_jaccard
            },
            'precision': {
                'per_class': precision_per_class,
                'mean': mean_precision
            },
            'recall': {
                'per_class': recall_per_class,
                'mean': mean_recall
            },
            'f_measure': {
                'per_class': f_measure_per_class,
                'mean': mean_f_measure
            },
            'specificity': {
                'per_class': specificity_per_class,
                'mean': mean_specificity
            },
            'class_counts': self.class_counts
        }


