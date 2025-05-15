import os
import csv
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Custom imports (adjust based on your project structure)
import cfg
import func_2d.function as function
from func_2d import utils as fn2dutils
from func_2d.inference_multi_utils import SAM2ImagePredictor

def find_annotated_image_paths(root_dir: str) -> List[str]:
    """
    Find paths to images with background mask
    
    Args:
        root_dir (str): Root directory to search for annotated images
    
    Returns:
        List of paths to annotated images
    """
    annotated_image_paths = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(root_dir):
        # Check if the current directory has frame.png and background.png
        if 'frame.png' in files and 'background.png' in files:
            frame_path = os.path.join(root, 'frame.png')
            annotated_image_paths.append(frame_path)
    
    return sorted(annotated_image_paths)

class FlexibleMultiClassMedSAM2Inference:
    def __init__(self, args, checkpoint_path):
        """
        Initialize the flexible multi-class MedSAM2 inference pipeline
        
        Args:
            args: Configuration arguments
            checkpoint_path: Path to the trained model checkpoint
        """
        # Set up device
        self.device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
        
        # Comprehensive class configuration
        self.class_config = {
            'background': {
                'index': 0,
                'color': [0, 0, 0],  # black
                'priority': 0
            },
            'solid': {
                'index': 1,
                'color': [1, 0, 0],  # red
                'priority': 2
            },
            'non-solid': {
                'index': 2,
                'color': [0, 0, 1],  # blue
                'priority': 1
            }
        }
        
        # Ordered list of class names for consistent processing
        self.class_names = ['background', 'solid', 'non-solid']
        
        # Model and checkpoint loading
        self.args = args
        self.net = fn2dutils.get_network(
            args, 
            args.net, 
            use_gpu=args.gpu, 
            gpu_device=self.device, 
            distribution=args.distributed
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model'])
        self.net.eval()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
    
    def inference_whole_image(self, image_path, confidence_threshold=0.5):
        """
        Perform flexible multi-class inference on a whole image
        
        Args:
            image_path (str): Path to the input image
            confidence_threshold (float): Threshold for mask confidence
        
        Returns:
            tuple: Processed masks, IoUs, original image path
        """
        try:
            # Load and preprocess image
            original_image = Image.open(image_path).convert('RGB')
            transformed_image = self.transform(original_image)
            
            # Create predictor
            predictor = SAM2ImagePredictor(self.net)
            predictor.set_image(original_image)
            
            # Get image dimensions
            h, w = np.array(original_image).shape[:2]
            whole_image_box = np.array([0, 0, h, w])
            
            # Predict masks
            masks, ious, low_res_masks = predictor.predict(
                box=whole_image_box,
                multimask_output=True,
                return_logits=True  # Keep logits for class-specific processing
            )
            
            # Post-process masks with confidence thresholding
            processed_masks = self._advanced_mask_processing(
                masks, 
                ious, 
                confidence_threshold
            )
            
            return processed_masks, ious, image_path
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], [], image_path
    
    def _advanced_mask_processing(self, masks, ious, confidence_threshold):
        """
        Advanced mask processing with flexible class handling
        
        Args:
            masks (np.ndarray): Raw predicted masks
            ious (np.ndarray): Intersection over Union predictions
            confidence_threshold (float): Minimum confidence to keep a mask
        
        Returns:
            list: Processed masks with comprehensive class information
        """
        processed_masks = []
        
        # Track which classes have been detected
        detected_classes = set()
        
        for i, (mask, iou) in enumerate(zip(masks, ious)):
            # Skip masks below confidence threshold
            if iou < confidence_threshold:
                continue
            
            # Determine class based on current index, with fallback
            class_name = self.class_names[min(i, len(self.class_names) - 1)]
            
            # Ensure unique class representation
            if class_name in detected_classes:
                # If class already detected, use priority to decide
                existing_mask = next(
                    (m for m in processed_masks if m['class_name'] == class_name), 
                    None
                )
                
                if existing_mask:
                    # Replace if new mask has higher IoU
                    if iou > existing_mask['iou']:
                        processed_masks.remove(existing_mask)
                    else:
                        continue
            
            # Add mask information
            mask_info = {
                'mask': mask,
                'iou': iou,
                'class_name': class_name,
                'class_color': self.class_config[class_name]['color'],
                'class_priority': self.class_config[class_name]['priority']
            }
            
            processed_masks.append(mask_info)
            detected_classes.add(class_name)
        
        # Sort masks by class priority (descending)
        processed_masks.sort(
            key=lambda x: self.class_config[x['class_name']]['priority'], 
            reverse=True
        )
        
        return processed_masks
    
    def batch_inference(
        self, 
        test_dir: str, 
        output_dir: str, 
        confidence_threshold: float = 0.5,
        ground_truth_dir: Optional[str] = None
    ):
        """
        Perform batch inference with optional ground truth comparison
        
        Args:
            test_dir (str): Directory containing test images
            output_dir (str): Directory to save results
            confidence_threshold (float): Confidence threshold for mask detection
            ground_truth_dir (Optional[str]): Directory with ground truth masks
        
        Returns:
            List of inference results
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        visualization_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Find annotated image paths
        image_paths = find_annotated_image_paths(test_dir)
        
        # Prepare CSV for results
        results_csv_path = os.path.join(output_dir, 'inference_results.csv')
        
        # Prepare results storage
        batch_results = []
        
        # Open CSV for writing results
        with open(results_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write comprehensive header
            csv_writer.writerow([
                'Image', 'Detected_Classes', 'Total_Masks', 
                'Background_IoU', 'Solid_IoU', 'NonSolid_IoU',
                'Ground_Truth_Metrics'
            ])
            
            # Process images
            for image_path in image_paths:
                # Perform inference
                processed_masks, ious, orig_path = self.inference_whole_image(
                    image_path, 
                    confidence_threshold
                )
                
                # Visualize and save results
                vis_path = self._visualize_and_save(
                    orig_path, 
                    processed_masks, 
                    visualization_dir
                )
                
                # Prepare result analysis
                detected_classes = [mask['class_name'] for mask in processed_masks]
                iou_by_class = {
                    'background': next((m['iou'] for m in processed_masks if m['class_name'] == 'background'), 0),
                    'solid': next((m['iou'] for m in processed_masks if m['class_name'] == 'solid'), 0),
                    'non-solid': next((m['iou'] for m in processed_masks if m['class_name'] == 'non-solid'), 0)
                }
                
                # Compare with ground truth masks if directory provided
                ground_truth_metrics = self._calculate_ground_truth_metrics(
                    orig_path, 
                    processed_masks, 
                    ground_truth_dir
                ) if ground_truth_dir else "No ground truth provided"
                
                # Record results in CSV
                csv_writer.writerow([
                    os.path.basename(orig_path),
                    ','.join(detected_classes),
                    len(processed_masks),
                    iou_by_class['background'],
                    iou_by_class['solid'],
                    iou_by_class['non-solid'],
                    str(ground_truth_metrics)
                ])
                
                # Store results
                batch_results.append({
                    'image_path': orig_path,
                    'visualization_path': vis_path,
                    'masks': processed_masks,
                    'detected_classes': detected_classes,
                    'ground_truth_metrics': ground_truth_metrics
                })
                
                print(f"Processed {os.path.basename(orig_path)}")
        
        print(f"\nBatch inference complete. Results saved to {output_dir}")
        print(f"Visualization saved to {visualization_dir}")
        print(f"CSV results saved to {results_csv_path}")
        
        return batch_results
    
    def _calculate_ground_truth_metrics(self, 
                                        pred_image_path, 
                                        processed_masks, 
                                        ground_truth_dir):
        """
        Calculate metrics by comparing predicted masks with ground truth
        
        Args:
            pred_image_path (str): Path to the predicted image
            processed_masks (List[Dict]): Predicted masks
            ground_truth_dir (str): Directory containing ground truth masks
        
        Returns:
            Dict with ground truth comparison metrics
        """
        # Find corresponding ground truth directory
        relative_path = os.path.relpath(os.path.dirname(pred_image_path), ground_truth_dir)
        
        metrics = {}
        classes = ['background', 'solid', 'non-solid']
        
        for class_name in classes:
            gt_mask_path = os.path.join(ground_truth_dir, relative_path, f'{class_name}.png')
            
            if os.path.exists(gt_mask_path):
                # Load ground truth mask
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                
                # Find corresponding predicted mask
                pred_mask = next((m['mask'] for m in processed_masks 
                                  if m['class_name'] == class_name), None)
                
                if pred_mask is not None:
                    # Convert to binary masks
                    pred_binary = (pred_mask > 0.5).astype(np.uint8)
                    gt_binary = (gt_mask > 0).astype(np.uint8)
                    
                    # Calculate IoU
                    intersection = np.logical_and(pred_binary, gt_binary)
                    union = np.logical_or(pred_binary, gt_binary)
                    iou = np.sum(intersection) / np.sum(union)
                    
                    metrics[class_name] = {
                        'IoU': iou,
                        'Exists_in_GT': True
                    }
                else:
                    metrics[class_name] = {
                        'IoU': 0,
                        'Exists_in_GT': True
                    }
            else:
                metrics[class_name] = {
                    'IoU': None,
                    'Exists_in_GT': False
                }
        
        return metrics
    
    def _visualize_and_save(
        self, 
        image_path: str, 
        processed_masks: List[Dict], 
        output_dir: str
    ) -> Optional[str]:
        """
        Visualize and save segmentation results with comprehensive information
        
        Args:
            image_path (str): Path to original image
            processed_masks (List[Dict]): Processed mask information
            output_dir (str): Directory to save visualizations
        
        Returns:
            Optional path to saved visualization
        """
        try:
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            
            # Create visualization
            plt.figure(figsize=(20, 6))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Mask overlay
            plt.subplot(1, 3, 2)
            plt.imshow(original_image)
            
            # Overlay masks with transparency
            for mask_info in processed_masks:
                mask = mask_info['mask']
                color = mask_info['class_color']
                plt.imshow(mask, color=color, alpha=0.5)
            
            plt.title('Multi-Class Segmentation')
            plt.axis('off')
            
            # Detailed mask information
            plt.subplot(1, 3, 3)
            plt.axis('off')
            info_text = "Detected Masks:\n"
            for mask_info in processed_masks:
                info_text += (
                    f"Class: {mask_info['class_name']}\n"
                    f"IoU: {mask_info['iou']:.4f}\n"
                )
            plt.text(0.1, 0.5, info_text, fontsize=10, 
                     verticalalignment='center')
            plt.title('Mask Details')
            
            # Save visualization
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f'segmentation_{filename}')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        except Exception as e:
            print(f"Error visualizing {image_path}: {e}")
            return None

def main():
    """
    Main function to run MedSAM2 Multi-Class Inference
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description='MedSAM2 Multi-Class Inference')
    
    # Input and output directories
    parser.add_argument('--test_dir', type=str, default='/shared/tesi-signani-data/dataset-segmentation/raw_dataset/test', 
                        help='Root directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results_multiclass', 
                        help='Directory to save inference results')
    
    # Model and inference parameters
    parser.add_argument('--checkpoint', type=str, 
                        default='logs/1/Model/latest_epoch.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--ground_truth_dir', type=str, default='/shared/tesi-signani-data/dataset-segmentation/raw_dataset/test',
                        help='Directory containing ground truth masks (optional)')
    parser.add_argument('--confidence_threshold', type=float, 
                        default=0.5, 
                        help='Confidence threshold for mask detection')
    
    # GPU and device settings
    parser.add_argument('--gpu_device', type=int, 
                        default=0, 
                        help='GPU device number')
    
    # Parse CLI arguments
    cli_args = parser.parse_args()
    
    # Load configuration
    args = cfg.parse_args()
    
    # Override GPU device if specified via CLI
    args.gpu_device = cli_args.gpu_device
    
    # Initialize inference
    inferencer = FlexibleMultiClassMedSAM2Inference(
        args, 
        cli_args.checkpoint
    )
    
    # Perform batch inference
    results = inferencer.batch_inference(
        test_dir=cli_args.test_dir, 
        output_dir=cli_args.output_dir, 
        confidence_threshold=cli_args.confidence_threshold,
        ground_truth_dir=cli_args.ground_truth_dir
    )
    
    # Optional: You can add further processing or analysis of results here
    return results

if __name__ == '__main__':
    main()