import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def gather_annotated_frames(input_root: Path) -> pd.DataFrame:
    """
    Walk input_root and for each subfolder containing frame.png plus
    at least one of background.png, solid.png, or nonsolid.png,
    record the paths (None if a given mask is missing).
    Also tracks mask presence to enable balancing.
    """
    records = []
    for dirpath, _, filenames in os.walk(input_root):
        files = {f.lower() for f in filenames}
        # must have the frame and at least one mask
        if 'frame.png' in files and any(m in files for m in ('background.png','solid.png','non-solid.png')):
            dirp = Path(dirpath)
            a = dirpath.split('/', -1)
            fan_path = Path(a[0] + '/' + a[1] + '/' + a[2] + '/' + a[3] + '/' + a[4] + '/' + a[5] + '/' + a[6]+ '/' + a[7] )
            #print(fan_path)
            #a
            rel = dirp.relative_to(input_root)
            
            # Check which masks are present
            has_background = 'background.png' in files
            has_solid = 'solid.png' in files
            has_nonsolid = 'non-solid.png' in files
            
            records.append({
                'clinical_case':  rel.parts[0],
                'item':           rel.as_posix(),
                'frame_path':     dirp / 'frame.png',
                'background_path':dirp / 'background.png' if has_background else None,
                'solid_path':     dirp / 'solid.png' if has_solid else None,
                'nonsolid_path':  dirp / 'non-solid.png' if has_nonsolid else None,
                'has_background': has_background,
                'has_solid':      has_solid,
                'has_nonsolid':   has_nonsolid,
                # optional fan:
                'fan_path': fan_path / 'fan.png'
            })
    return pd.DataFrame(records)


def create_visualizations(df, output_dir, hist_df=None):
    """
    Create and save visualizations of the dataset distribution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})
    
    # 1. Distribution of frames per clinical case
    plt.figure(figsize=(12, 8))
    frame_counts = df.groupby('clinical_case').size()
    sns.histplot(frame_counts, kde=True)
    plt.title('Distribution of Frames per Clinical Case')
    plt.xlabel('Number of Frames')
    plt.ylabel('Count of Clinical Cases')
    plt.axvline(frame_counts.mean(), color='r', linestyle='--', 
                label=f'Mean: {frame_counts.mean():.2f}')
    plt.axvline(frame_counts.median(), color='g', linestyle='-', 
                label=f'Median: {frame_counts.median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'frames_per_case_distribution.png')
    plt.close()
    
    # 2. Mask type distribution
    plt.figure(figsize=(10, 6))
    mask_counts = pd.DataFrame({
        'Background': df['has_background'].sum(),
        'Solid': df['has_solid'].sum(),
        'Non-solid': df['has_nonsolid'].sum(),
        'Fan': df['has_fan'].sum() if 'has_fan' in df.columns else 0
    }, index=['Count'])
    mask_counts = mask_counts.T
    ax = sns.barplot(x=mask_counts.index, y=mask_counts['Count'])
    plt.title('Distribution of Mask Types')
    plt.ylabel('Count')
    plt.xlabel('Mask Type')
    
    # Add percentages on top of bars
    total = len(df)
    for i, v in enumerate(mask_counts['Count']):
        ax.text(i, v + 5, f"{v/total*100:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mask_type_distribution.png')
    plt.close()
    
    # 3. Combinations of masks
    plt.figure(figsize=(14, 8))
    mask_combinations = df.groupby(['has_background', 'has_solid', 'has_nonsolid']).size().reset_index()
    mask_combinations['mask_combo'] = mask_combinations.apply(
        lambda x: f"BG: {'✓' if x['has_background'] else '✗'}, "
                  f"Solid: {'✓' if x['has_solid'] else '✗'}, "
                  f"Non-solid: {'✓' if x['has_nonsolid'] else '✗'}", axis=1)
    
    combo_counts = mask_combinations[0].values
    combo_labels = mask_combinations['mask_combo'].values
    
    # Sort by count
    sorted_indices = np.argsort(combo_counts)[::-1]
    combo_counts = combo_counts[sorted_indices]
    combo_labels = combo_labels[sorted_indices]
    
    ax = sns.barplot(x=combo_labels, y=combo_counts)
    plt.title('Combinations of Mask Types')
    plt.ylabel('Count')
    plt.xlabel('Mask Combination')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentages on top of bars
    for i, v in enumerate(combo_counts):
        ax.text(i, v + 5, f"{v/total*100:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mask_combinations.png')
    plt.close()
    
    # 4. If histological data is available, show distribution
    if hist_df is not None:
        try:
            case_hist = df[['clinical_case']].drop_duplicates()
            case_hist = case_hist.merge(hist_df[['clinical_case', 'histological']], on='clinical_case', how='left')
            case_hist['histological'].fillna('unknown', inplace=True)
            
            plt.figure(figsize=(12, 8))
            hist_counts = case_hist['histological'].value_counts()
            ax = sns.barplot(x=hist_counts.index, y=hist_counts.values)
            plt.title('Distribution of Histological Types')
            plt.ylabel('Count of Clinical Cases')
            plt.xlabel('Histological Type')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentages on top of bars
            total_cases = len(case_hist)
            for i, v in enumerate(hist_counts.values):
                ax.text(i, v + 0.5, f"{v/total_cases*100:.1f}%", ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'histological_distribution.png')
            plt.close()
            
            # 5. Frames per histological type
            plt.figure(figsize=(14, 8))
            frames_by_hist = df.merge(case_hist, on='clinical_case')
            hist_frame_counts = frames_by_hist.groupby('histological').size()
            ax = sns.barplot(x=hist_frame_counts.index, y=hist_frame_counts.values)
            plt.title('Number of Frames by Histological Type')
            plt.ylabel('Count of Frames')
            plt.xlabel('Histological Type')
            plt.xticks(rotation=45, ha='right')
            
            '''
            # Add percentages on top of bars
            for i, v in enumerate(hist_frame_counts.values):
                ax.text(i, v + 5, f"{v/total*100:.1f}%", ha='center')
            '''
            
            plt.tight_layout()
            plt.savefig(output_dir / 'frames_by_histological_type.png')
            plt.close()
            
            # 6. Mask distribution by histological type
            plt.figure(figsize=(16, 10))
            mask_by_hist = frames_by_hist.groupby('histological').agg({
                'has_background': 'sum',
                'has_solid': 'sum',
                'has_nonsolid': 'sum'
            }).reset_index()
            
            mask_by_hist_melted = pd.melt(
                mask_by_hist, 
                id_vars=['histological'],
                value_vars=['has_background', 'has_solid', 'has_nonsolid'],
                var_name='Mask Type',
                value_name='Count'
            )
            mask_by_hist_melted['Mask Type'] = mask_by_hist_melted['Mask Type'].map({
                'has_background': 'Background',
                'has_solid': 'Solid',
                'has_nonsolid': 'Non-solid'
            })
            
            sns.barplot(x='histological', y='Count', hue='Mask Type', data=mask_by_hist_melted)
            plt.title('Distribution of Mask Types by Histological Category')
            plt.ylabel('Count')
            plt.xlabel('Histological Type')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Mask Type')
            plt.tight_layout()
            plt.savefig(output_dir / 'mask_by_histological.png')
            plt.close()
            
        except Exception as e:
            print(f"Could not create histological visualizations: {e}")
    
    return output_dir


def evaluate_fold_balance(folds, df):
    """
    Evaluate the balance of solid/non-solid masks across folds.
    Returns a DataFrame with metrics for each fold.
    Includes robust error handling for edge cases.
    """
    results = []
    
    try:
        # Check if required columns exist
        required_cols = ['has_solid', 'has_nonsolid']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Missing column {col}. Adding dummy values.")
                df[col] = False  # Default to no masks if column is missing
        
        # Calculate overall dataset statistics
        total_frames = max(len(df), 1)  # Avoid division by zero
        total_solid = df['has_solid'].sum()
        total_nonsolid = df['has_nonsolid'].sum()
        
        overall_solid_ratio = total_solid / total_frames
        overall_nonsolid_ratio = total_nonsolid / total_frames
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            try:
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                # Calculate metrics
                train_frames = max(len(train_df), 1)  # Avoid division by zero
                val_frames = max(len(val_df), 1)      # Avoid division by zero
                
                train_solid = train_df['has_solid'].sum()
                train_nonsolid = train_df['has_nonsolid'].sum()
                
                val_solid = val_df['has_solid'].sum()
                val_nonsolid = val_df['has_nonsolid'].sum()
                
                train_solid_ratio = train_solid / train_frames
                train_nonsolid_ratio = train_nonsolid / train_frames
                
                val_solid_ratio = val_solid / val_frames
                val_nonsolid_ratio = val_nonsolid / val_frames
                
                # Calculate imbalance (difference from overall dataset)
                train_solid_imbalance = abs(train_solid_ratio - overall_solid_ratio)
                train_nonsolid_imbalance = abs(train_nonsolid_ratio - overall_nonsolid_ratio)
                
                val_solid_imbalance = abs(val_solid_ratio - overall_solid_ratio)
                val_nonsolid_imbalance = abs(val_nonsolid_ratio - overall_nonsolid_ratio)
                
                # Total imbalance score (lower is better)
                total_imbalance = (
                    train_solid_imbalance + train_nonsolid_imbalance +
                    val_solid_imbalance + val_nonsolid_imbalance
                )
                
                results.append({
                    'fold': fold_idx,
                    'train_frames': train_frames,
                    'val_frames': val_frames,
                    'train_solid': train_solid,
                    'train_nonsolid': train_nonsolid,
                    'val_solid': val_solid,
                    'val_nonsolid': val_nonsolid,
                    'train_solid_ratio': train_solid_ratio,
                    'train_nonsolid_ratio': train_nonsolid_ratio,
                    'val_solid_ratio': val_solid_ratio,
                    'val_nonsolid_ratio': val_nonsolid_ratio,
                    'imbalance_score': total_imbalance
                })
            except Exception as e:
                print(f"Warning: Error processing fold {fold_idx}: {e}")
                # Add a placeholder record with defaults to ensure we have something for each fold
                results.append({
                    'fold': fold_idx,
                    'train_frames': len(train_idx),
                    'val_frames': len(val_idx),
                    'train_solid': 0,
                    'train_nonsolid': 0,
                    'val_solid': 0,
                    'val_nonsolid': 0,
                    'train_solid_ratio': 0,
                    'train_nonsolid_ratio': 0,
                    'val_solid_ratio': 0,
                    'val_nonsolid_ratio': 0,
                    'imbalance_score': float('inf')  # Mark as bad fold
                })
    except Exception as e:
        print(f"Error in fold evaluation: {e}")
        # Return minimal DataFrame if overall evaluation fails
        for fold_idx in range(len(folds)):
            results.append({
                'fold': fold_idx,
                'train_frames': 0,
                'val_frames': 0,
                'train_solid': 0,
                'train_nonsolid': 0,
                'val_solid': 0,
                'val_nonsolid': 0,
                'train_solid_ratio': 0,
                'train_nonsolid_ratio': 0,
                'val_solid_ratio': 0,
                'val_nonsolid_ratio': 0,
                'imbalance_score': float('inf')
            })
    
    result_df = pd.DataFrame(results)
    
    # Additional safety check - replace any NaN values
    result_df.fillna({'imbalance_score': float('inf')}, inplace=True)
    for col in result_df.columns:
        if result_df[col].dtype in (np.float64, np.float32):
            result_df[col].fillna(0, inplace=True)
    
    return result_df


def make_stratified_group_folds(
    input_root: Path,
    output_root: Path,
    csv_root: Path,
    n_splits: int = 5,
    random_state: int = 42,
    n_bins: int = 4,
    max_attempts: int = 10
):
    # Load and prepare data
    df = gather_annotated_frames(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create overall dataset visualizations
    try:
        hist_df = pd.read_csv(csv_root)
        vis_dir = create_visualizations(df, output_root / 'dataset_analysis', hist_df)
        print(f"Dataset visualizations saved to {vis_dir}")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
        hist_df = None
    
    # Build case-level metadata for stratification
    case_df = pd.DataFrame({'clinical_case': df['clinical_case'].unique()})
    
    # Add histological data if available
    if hist_df is not None:
        case_df = case_df.merge(hist_df[['clinical_case', 'histological']], on='clinical_case', how='left')
        case_df['histological'].fillna('unknown', inplace=True)
    else:
        case_df['histological'] = 'unknown'
    
    # Count frames per case and bin them
    frame_counts = df.groupby('clinical_case').size().rename('frame_count')
    case_df = case_df.join(frame_counts, on='clinical_case')
    
    # Make the binning more robust to handle cases with too few unique values
    try:
        case_df['count_bin'] = pd.qcut(
            case_df['frame_count'], q=n_bins, labels=False, duplicates='drop'
        )
    except ValueError:
        # Fall back to fewer bins if we can't create n_bins
        try:
            actual_bins = min(n_bins, len(case_df['frame_count'].unique()) - 1)
            if actual_bins <= 1:
                # If we can't even do 2 bins, just use a binary split at median
                median_count = case_df['frame_count'].median()
                case_df['count_bin'] = np.where(
                    case_df['frame_count'] <= median_count, 0, 1
                )
                print(f"Warning: Not enough unique frame counts for {n_bins} bins. Using binary split.")
            else:
                case_df['count_bin'] = pd.qcut(
                    case_df['frame_count'], q=actual_bins, labels=False, duplicates='drop'
                )
                print(f"Warning: Reduced bins from {n_bins} to {actual_bins} due to limited unique values.")
        except ValueError:
            # As a last resort, just assign all to same bin
            case_df['count_bin'] = 0
            print("Warning: Could not create multiple bins for frame counts. Using a single bin.")
    
    # Add mask presence information per case
    mask_presence = df.groupby('clinical_case').agg({
        'has_solid': 'mean',
        'has_nonsolid': 'mean'
    })
    case_df = case_df.join(mask_presence, on='clinical_case')
    
    # Create stratification labels - with more robust binning that handles edge cases
    # For solid masks
    try:
        # First try qcut with 2 bins
        case_df['solid_bin'] = pd.qcut(
            case_df['has_solid'], q=2, labels=['low_solid', 'high_solid'], duplicates='drop'
        )
    except ValueError:
        # If that fails, use the median as a threshold
        median_solid = case_df['has_solid'].median()
        case_df['solid_bin'] = np.where(
            case_df['has_solid'] <= median_solid, 'low_solid', 'high_solid'
        )
    
    # For non-solid masks
    try:
        # First try qcut with 2 bins
        case_df['nonsolid_bin'] = pd.qcut(
            case_df['has_nonsolid'], q=2, labels=['low_nonsolid', 'high_nonsolid'], duplicates='drop'
        )
    except ValueError:
        # If that fails, use the median as a threshold
        median_nonsolid = case_df['has_nonsolid'].median()
        case_df['nonsolid_bin'] = np.where(
            case_df['has_nonsolid'] <= median_nonsolid, 'low_nonsolid', 'high_nonsolid'
        )
    
    # Combined stratification label
    # Make sure all columns exist and are properly formatted
    required_cols = ['histological', 'count_bin', 'solid_bin', 'nonsolid_bin']
    for col in required_cols:
        if col not in case_df.columns:
            print(f"Warning: Missing column {col}. Using placeholder.")
            case_df[col] = 'unknown'
    
    # Convert all components to string to avoid type issues
    case_df['strat_label'] = (
        case_df['histological'].astype(str) + '_' +
        'bin' + case_df['count_bin'].astype(str) + '_' +
        case_df['solid_bin'].astype(str) + '_' +
        case_df['nonsolid_bin'].astype(str)
    )
    
    # Handle NaN in stratification labels (can happen with qcut)
    case_df['strat_label'] = case_df['strat_label'].fillna('unknown')
    case_df.drop_duplicates(subset='clinical_case', inplace=True)
    
    # Prepare y and groups for StratifiedGroupKFold
    y = df['clinical_case'].map(case_df.set_index('clinical_case')['strat_label'])
    groups = df['clinical_case']
    
    # Try multiple random states to find the most balanced fold distribution
    best_folds = None
    best_balance_score = float('inf')
    best_seed = random_state
    
    for attempt in range(max_attempts):
        current_seed = random_state + attempt
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=current_seed
        )
        
        # Generate folds
        current_folds = list(sgkf.split(df, y=y, groups=groups))
        
        # Evaluate fold balance
        balance_df = evaluate_fold_balance(current_folds, df)
        current_balance_score = balance_df['imbalance_score'].mean()
        
        if current_balance_score < best_balance_score:
            best_balance_score = current_balance_score
            best_folds = current_folds
            best_seed = current_seed
    
    print(f"Best seed found: {best_seed} with balance score: {best_balance_score:.4f}")
    
    # Get final statistics and generate visualizations
    balance_df = evaluate_fold_balance(best_folds, df)
    print("\nFold Distribution Statistics:")
    print(balance_df[['fold', 'train_frames', 'val_frames', 'train_solid_ratio', 
                      'train_nonsolid_ratio', 'val_solid_ratio', 'val_nonsolid_ratio', 
                      'imbalance_score']])
    
    # Visualize fold balance
    plt.figure(figsize=(14, 8))
    
    # Plot solid mask ratio
    plt.subplot(1, 2, 1)
    plt.axhline(y=df['has_solid'].mean(), color='r', linestyle='--', 
                label=f'Overall: {df["has_solid"].mean():.2f}')
    
    sns.barplot(x='fold', y='train_solid_ratio', data=balance_df, color='blue', alpha=0.7, label='Train')
    sns.barplot(x='fold', y='val_solid_ratio', data=balance_df, color='green', alpha=0.7, label='Validation')
    
    plt.title('Solid Mask Ratio by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Solid Mask Ratio')
    plt.legend()
    
    # Plot non-solid mask ratio
    plt.subplot(1, 2, 2)
    plt.axhline(y=df['has_nonsolid'].mean(), color='r', linestyle='--', 
                label=f'Overall: {df["has_nonsolid"].mean():.2f}')
    
    sns.barplot(x='fold', y='train_nonsolid_ratio', data=balance_df, color='blue', alpha=0.7, label='Train')
    sns.barplot(x='fold', y='val_nonsolid_ratio', data=balance_df, color='green', alpha=0.7, label='Validation')
    
    plt.title('Non-solid Mask Ratio by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Non-solid Mask Ratio')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_root / 'fold_balance_analysis.png')
    plt.close()
    
    # Create table of fold statistics
    fold_stats = balance_df.copy()
    fold_stats['train_solid_pct'] = fold_stats['train_solid_ratio'] * 100
    fold_stats['train_nonsolid_pct'] = fold_stats['train_nonsolid_ratio'] * 100
    fold_stats['val_solid_pct'] = fold_stats['val_solid_ratio'] * 100
    fold_stats['val_nonsolid_pct'] = fold_stats['val_nonsolid_ratio'] * 100
    
    plt.figure(figsize=(12, n_splits * 0.8 + 2))
    plt.axis('off')
    
    col_labels = ['Fold', 'Train Frames', 'Val Frames', 
                  'Train Solid %', 'Train Non-solid %',
                  'Val Solid %', 'Val Non-solid %', 
                  'Imbalance Score']
    
    table_data = fold_stats[['fold', 'train_frames', 'val_frames', 
                           'train_solid_pct', 'train_nonsolid_pct',
                           'val_solid_pct', 'val_nonsolid_pct', 
                           'imbalance_score']].values
    
    # Format percentages
    formatted_data = []
    for row in table_data:
        formatted_row = [
            f"{row[0]:.0f}",                   # fold 
            f"{row[1]:.0f}",                   # train_frames
            f"{row[2]:.0f}",                   # val_frames
            f"{row[3]:.1f}%",                  # train_solid_pct
            f"{row[4]:.1f}%",                  # train_nonsolid_pct
            f"{row[5]:.1f}%",                  # val_solid_pct
            f"{row[6]:.1f}%",                  # val_nonsolid_pct
            f"{row[7]:.4f}"                    # imbalance_score
        ]
        formatted_data.append(formatted_row)
    
    table = plt.table(
        cellText=formatted_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.08, 0.12, 0.12, 0.13, 0.16, 0.13, 0.16, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Fold Statistics Summary', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_root / 'fold_statistics_table.png')
    plt.close()
    
    # Copy out files for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(best_folds):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        print(f"\n=== Fold {fold_idx} ===")
        for name, split_df in (('TRAIN', train_df), ('VAL', val_df)):
            hist_prop = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['histological'])
                .value_counts(normalize=True)
            )
            bin_prop = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['count_bin'])
                .value_counts(normalize=True)
            )
            
            # Calculate mask distributions
            solid_ratio = split_df['has_solid'].mean()
            nonsolid_ratio = split_df['has_nonsolid'].mean()
            
            print(f"{name} hist proportions:\n{hist_prop.to_dict()}")
            print(f"{name} size-bin proportions:\n{bin_prop.to_dict()}")
            print(f"{name} solid mask ratio: {solid_ratio:.2f}")
            print(f"{name} non-solid mask ratio: {nonsolid_ratio:.2f}\n")
        
        # Copy out files for this fold
        fold_dir = output_root / f'fold_{fold_idx}'
        for split_name, subset in [('train', train_df), ('val', val_df)]:
            tgt = fold_dir / split_name
            for _, row in subset.iterrows():
                dest = tgt / row['clinical_case'] / Path(row['item'])
                dest.mkdir(parents=True, exist_ok=True)
                
                # copy the frame
                shutil.copy2(row['frame_path'], dest / 'frame.png')
                
                # copy each mask if present
                if row['background_path'] is not None:
                    shutil.copy2(row['background_path'], dest / 'background.png')
                if row['solid_path'] is not None:
                    shutil.copy2(row['solid_path'], dest / 'solid.png')
                if row['nonsolid_path'] is not None:
                    shutil.copy2(row['nonsolid_path'], dest / 'non-solid.png')
                
                a = str(dest).split('/', -1)
                fan_path = Path(a[0] + '/' + a[1] + '/' + a[2] + '/' + a[3] + '/' + a[4] + '/' + a[5])
                if 'fan.png' not in os.walk(fan_path):
                    shutil.copy2(row['fan_path'], fan_path / 'fan.png')
    
    # Generate summary report
    print(f"\nData split complete. Files copied to {output_root}")
    print(f"Dataset analysis visualizations saved to {output_root / 'dataset_analysis'}")
    print(f"Fold balance visualizations saved to {output_root}")
    
    # Save metadata about the splits for future reference
    metadata = {
        'total_frames': len(df),
        'total_cases': len(case_df),
        'total_solid_masks': df['has_solid'].sum(),
        'total_nonsolid_masks': df['has_nonsolid'].sum(),
        'solid_ratio': df['has_solid'].mean(),
        'nonsolid_ratio': df['has_nonsolid'].mean(),
        'seed_used': best_seed,
        'n_splits': n_splits,
        'n_bins': n_bins,
        'balance_score': best_balance_score
    }
    
    pd.DataFrame([metadata]).to_csv(output_root / 'split_metadata.csv', index=False)
    balance_df.to_csv(output_root / 'fold_statistics.csv', index=False)
    
    return output_root


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root',  type=Path, default=Path('/shared/tesi-signani-data/dataset-segmentation/raw_dataset/train'))
    parser.add_argument('--output_root', type=Path, default=Path('Multiclass_Folds'))
    parser.add_argument('--csv_root',    type=Path, default=Path('clinical-cases-metadata-holsbeke.csv'))
    parser.add_argument('--splits',      type=int,  default=5)
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--attempts',    type=int,  default=10,
                        help='Number of random seeds to try for optimal balance')
    args = parser.parse_args()
    
    make_stratified_group_folds(
        input_root=args.input_root,
        output_root=args.output_root,
        csv_root=args.csv_root,
        n_splits=args.splits,
        random_state=args.seed,
        n_bins=4,
        max_attempts=args.attempts
    )