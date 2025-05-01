import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def gather_annotated_frames(input_root: Path) -> pd.DataFrame:
    """
    Find all pairs of (frame.png, background.png) under input_root,
    grouped by clinical_case (top-level folder).

    Returns:
        DataFrame with columns [clinical_case, item, frame_path, gt_path]
    """
    records = []
    for dirpath, _, filenames in os.walk(input_root):
        files = {f.lower() for f in filenames}
        if 'frame.png' in files and 'background.png' in files:
            dirp = Path(dirpath)
            rel = dirp.relative_to(input_root)
            records.append({
                'clinical_case': rel.parts[0],
                'item': rel.as_posix(),
                'frame_path': dirp / 'frame.png',
                'gt_path':    dirp / 'background.png'
            })
    return pd.DataFrame(records)


def make_stratified_group_folds(
    input_root: Path,
    output_root: Path,
    csv_root : Path,
    n_splits: int = 5,
    random_state: int = 42,
    n_bins: int = 4
):
    # 1. Gather frames
    df = gather_annotated_frames(input_root)

    # 2. Unique cases
    case_df = pd.DataFrame({'clinical_case': df['clinical_case'].unique()})

    # 3. Load your external histology CSV
    hist_df = pd.read_csv(csv_root)  
    # should have columns: clinical_case, histological

    # 4. Merge & fill missing
    case_df = case_df.merge(hist_df, on='clinical_case', how='left')
    case_df['histological'] = case_df['histological'].fillna('unknown')

    # 5. Count frames per case
    frame_counts = df.groupby('clinical_case').size().rename('frame_count')
    case_df = case_df.join(frame_counts, on='clinical_case')

    # 6. Bin frame counts into q quantiles
    case_df['count_bin'] = pd.qcut(
        case_df['frame_count'], 
        q=n_bins, 
        labels=False, 
        duplicates='drop'
    )

    # 7. Create the combined stratification label
    case_df['strat_label'] = (
        case_df['histological'].astype(str)
    + '_bin' 
    + case_df['count_bin'].astype(str)
    )

    case_df = case_df.drop_duplicates(subset='clinical_case')

    y = df['clinical_case'].map(case_df.set_index('clinical_case')['strat_label'])
    
    groups = df['clinical_case']

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(df, y=y, groups=groups)
    ):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        # Quick check: hist + frame‐count‐bin distributions
        print(f"=== Fold {fold_idx} ===")
        for name, split_df in (('TRAIN', train_df), ('VAL', val_df)):
            dist = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['histological'])
                .value_counts(normalize=True)
            )
            bin_dist = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['count_bin'])
                .value_counts(normalize=True)
            )
            print(f"{name} hist proportions:\n{dist.to_dict()}")
            print(f"{name} size‐bin proportions:\n{bin_dist.to_dict()}\n")

        # Copy files
        fold_dir = output_root / f'fold_{fold_idx}'
        for split_name, subset in [('train', train_df), ('val', val_df)]:
            tgt = fold_dir / split_name
            for _, row in subset.iterrows():
                dest = tgt / row['clinical_case'] / Path(row['item'])
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(row['frame_path'], dest / 'frame.png')
                shutil.copy2(row['gt_path'],    dest / 'background.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    input_root = Path('Preliminary-data')
    output_root = Path('Folds')
    csv_root = Path('clinical-cases-metadata-holsbeke.csv')
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    make_stratified_group_folds(
        input_root=input_root,
        output_root=output_root,
        n_splits=args.splits,
        random_state=args.seed,
        csv_root=csv_root,
        n_bins=4
    )


