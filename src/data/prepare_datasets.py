""" 
Age Estimation - Data Preparation
"""

# ===============================================================================
# IMPORTS
# ===============================================================================
import os
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict
import argparse
from scipy.io import loadmat


# ===============================================================================
# UTILS & IMAGE PROCESSING FUNCTIONS
# ===============================================================================

def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 grayscale, handling normalization and NaNs."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels in (1, 3, 4):
            img = arr[:, :, :3] if channels >= 3 else arr[:, :, 0]
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    
    img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

    if img.dtype != np.uint8:
        maxv = float(np.nanmax(img)) if img.size else 0.0
        if maxv <= 1.0:
            img = (img * 255.0).astype(np.uint8) # Scale 0-1 to 0-255
        elif maxv > 255.0:
            img = np.clip(img, 0, 255).astype(np.uint8) # Clip and scale if values are unexpectedly high but not uint8
        else:
            img = img.astype(np.uint8)
    return img


def image_to_gray_pixels(image_array: np.ndarray, size: int = 224) -> str:
    """
    Convert image to grayscale and return as space-separated pixel string.
    Uses LANCZOS for better quality resizing on facial features.
    """
    try:
        img_u8 = to_uint8_image(image_array)
        pil_img = Image.fromarray(img_u8, mode='L').resize((size, size), Image.LANCZOS)
        return " ".join(map(str, np.array(pil_img, dtype=np.uint8).flatten()))
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")


def log_line(line: str, logs: list[str], r = False, disp = True) -> None:
    """Append message to logs list and optionally print."""
    if not isinstance(logs, list): logs = []
    logs.append(line)
    if disp: print(line, end='\r' if r else '\n') 

# ===============================================================================
# DATASET SPECIFIC PARSERS
# Each function returns : (dataset, categories) where dataset is a list of tuples and categories is a list of unique categories for splitting or None if not applicable.
# ===============================================================================


def parse_agedb(root: Path, size: int, children: bool, logs: List[str]) -> Tuple[List[Tuple], List[str]]:
    dataset = []
    categories = []
    # Implement parsing logic for AgeDB dataset
    files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for i, filename in enumerate(files):
        try:
            # Expected format: id_cat_age_extension (e.g., 0001_01_25_001.jpg)
            parts = filename.split('_')
            if len(parts) < 3: continue

            img_id, cat, age = parts[0], parts[1], parts[2]
            if children and (int(age) < 0 or int(age) > 16): continue  # Skip if filtering for children
            if cat not in categories: categories.append(cat)

            img = Image.open(root / filename).convert("L")
            pixels = image_to_gray_pixels(np.array(img), size=size)
            dataset.append((img_id, cat, age, pixels))

            log_line(f"[AgeDB] |{'█'*int((i+1)/len(files)*20):<20}| Processing image {i+1}/{len(files)} [{(i+1)/len(files)*100:.1f}%]     ", logs, r=True)
        except Exception as e:
            log_line(f"[AgeDB] Error processing {filename}: {str(e)}", logs)

    return dataset, categories


def parse_utkface(root: Path, size: int, children: bool, logs: List[str]) -> Tuple[List[Tuple], List[str]]:
    dataset = []
    categories = []  # UTKFace does not have categories for splitting, but we keep the structure for consistency

    files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for i, filename in enumerate(files):
        try:
            # Expected format: age_gender_(race_)date.jpg (e.g., 25_0_1_20170116174525125.jpg)
            parts = filename.split('_')
            if len(parts) < 3: continue

            age = parts[0]
            if children and (int(age) < 0 or int(age) > 16): continue  # Skip if filtering for children
            img_id = parts[3].split('.')[0]  # Use the date part as a unique identifier

            img = Image.open(root / filename).convert("L")
            pixels = image_to_gray_pixels(np.array(img), size=size)

            dataset.append((img_id, None, age, pixels))  # No category for UTKFace
            log_line(f"[UTKFace] |{'█'*int((i+1)/len(files)*20):<20}| Processing image {i+1}/{len(files)} [{(i+1)/len(files)*100:.1f}%]     ", logs, r=True)
        except Exception as e:
            log_line(f"[UTKFace] Error processing {filename}: {str(e)}", logs)
    
    return dataset, categories


def parse_imdb_wiki(root: Path, size: int, children: bool, logs: List[str]) -> Tuple[List[Tuple], List[str]]:
    dataset = []
    categories = []

    data_name = str(root).split('/')[-1].lower()

    mat_path = root / f"{data_name}.mat"
    if not mat_path.exists():
        log_line(f"[{data_name.upper()}] Error: {data_name}.mat file not found in {root}", logs)
        return dataset, categories
    
    try:
        mat = loadmat(mat_path)
        data = mat[f"{data_name}"]
        photos_taken = data[0][0][1][0]
        paths = data[0][0][2][0]
        names = data[0][0][4][0]
        face_locations = data[0][0][5][0]
        face_scores1 = data[0][0][6][0]
        face_scores2 = data[0][0][7][0]
    except Exception as e:
        log_line(f"[{data_name.upper()}] Error loading {data_name}.mat: {str(e)}", logs)
        return dataset, categories
    
    for i in range(len(paths)):
        try:
            if face_scores1[i] == -np.inf or not np.isnan(face_scores2[i]): continue # Skip if no face or multiple faces detected

            full_path = paths[i][0]
            img_path = root / full_path
            if not img_path.exists(): continue

            # Crop to face bounding box
            img = Image.open(img_path).convert("L")

            loc = face_locations[i][0]
            left, top, right, bottom = int(loc[0]), int(loc[1]), int(loc[2])+1, int(loc[3])+1
            w, h = img.size
            left, top, right, bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
            if left >= right or top >= bottom: continue  # Invalid bounding box

            crop = img.crop((left, top, right, bottom))
            pixels = image_to_gray_pixels(np.array(crop), size=size)

            # Extract age from photo taken and full path
            fname = os.path.basename(full_path)
            parts = fname.split('_')
            if len(parts) < 2: continue
            yob = int(parts[1 if data_name == "wiki" else 2].split('-')[0]) 
            age = photos_taken[i] - yob

            if children and (int(age) < 0 or int(age) > 16): continue  # Skip if filtering for children

            name = names[i][0] if names[i].size > 0 else "unknown"
            if name not in categories: categories.append(name)

            dataset.append((parts[0], name, str(age), pixels))
            log_line(f"[{data_name.upper()}] |{'█'*int((i+1)/len(paths)*20):<20}| Processing image {i+1}/{len(paths)} [{(i+1)/len(paths)*100:.1f}%]     ", logs, r=True)
        except Exception as e:
            log_line(f"[{data_name.upper()}] Error processing {full_path}: {str(e)}", logs)

    return dataset, categories


# ===============================================================================
# CORE FUNCTIONS (Split and Save)
# ===============================================================================



def train_val_test_split(
    dataset: List[Tuple], 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.20, 
    categories: Optional[List[str]] = None
) -> Tuple[List, List, List]:
    """
    Split dataset into train, val, and test sets.
    If categories is provided, splits by category to ensure all samples 
    from the same category stay together (prevents data leakage).
    """
    if not dataset: return [], [], []

    if categories:
        # Ensure unique categories and shuffle them
        unique_cats = list(set(c for c in categories if c is not None))
        random.shuffle(unique_cats)

        n = len(unique_cats)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_cats = set(unique_cats[:train_end])
        val_cats = set(unique_cats[train_end:val_end])
        test_cats = set(unique_cats[val_end:])

        train_set = [sample for sample in dataset if sample[1] in train_cats]
        val_set = [sample for sample in dataset if sample[1] in val_cats]
        test_set = [sample for sample in dataset if sample[1] in test_cats]
    
    else :
        # Random split of individual samples (risk of leakage if samples share IDs)
        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        n = len(dataset_shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set, val_set , test_set = dataset_shuffled[:train_end], dataset_shuffled[train_end:val_end], dataset_shuffled[val_end:]

    return train_set, val_set, test_set
    

def save_splits(name: str, size: int, children: bool, dataset: List[Tuple], categories: List[str], output_dir: Path, logs: list[str], dry_run: bool = False) -> None:
    """Save dataset to CSV with specified columns."""
    # Split dataset
    train, val, test = train_val_test_split(dataset=dataset, categories=categories if categories else None)

    splits = {"train": train, "val": val, "test": test}

    # Save to CSV
    for split_name, data in splits.items():
        if not data: continue

        # Convert to DataFrame and save, "cat" column is empty or None if categories are not applicable
        df = pd.DataFrame(data, columns=["id", "cat", "age", "pixels"])

        out_path = output_dir / f"{split_name}-{'children-' if children else ''}{name.lower()}{size}.csv"
        if not dry_run:
            df.to_csv(out_path, index=False)
        else:
            log_line(f"[{name.upper()}] DRY-RUN: would save {len(df)} rows to {out_path}", logs)
            continue
        log_line(f"[{name.upper()}] Saved {len(df)} rows to {out_path}", logs)


# ===============================================================================
# MAIN ORChESTRATION FUNCTION
# ===============================================================================

DATASET_CONFIG: Dict[str, Dict] = {
    "agedb": {
        "folder": "AgeDB",
        "parser": parse_agedb,
        "has_categories": True 
    },
    "utkface": {
        "folder": "UTKFace",
        "parser": parse_utkface,
        "has_categories": False
    },
    "wiki": {
        "folder": "Wiki",
        "parser": parse_imdb_wiki,
        "has_categories": True
    },
    "imdb": {
        "folder": "IMDb",
        "parser": parse_imdb_wiki,
        "has_categories": True
    }
}


def prepare_dataset(name: str, size: int, children: bool, data_dir: Path, output_dir: Path, dry_run: bool, logs: list[str]) -> None:
    config = DATASET_CONFIG.get(name.lower())
    if not config:
        log_line(f"[{name.upper()}] Error: No configuration found for dataset '{name}'.", logs)
        return

    root = data_dir / config["folder"]
    if not root.exists():
        log_line(f"[{name.upper()}] Directory not found: {root}", logs)
        return

    log_line(f"[{name.upper()}] Starting dataset preparation...", logs)
    dataset, categories = config["parser"](root, size, children, logs)

    if not dataset:
        log_line(f"[{name.upper()}] No valid data found. Skipping.", logs)
        return

    save_splits(name, size, children, dataset, categories, output_dir, logs=logs, dry_run=dry_run)
    log_line(f"[{name.upper()}] Dataset preparation completed. Total samples: {len(dataset)}", logs)
 


# --- 3. EXECUTION ---

if __name__ == "__main__":
    # CONFIGURATION
    parser = argparse.ArgumentParser(description="Prepare dataset for age estimation.")
    parser.add_argument("--dataset", type=str, default="AgeDB", choices=["all", "AgeDB", "UTKFace", "Wiki", "IMDb"], help="Dataset to prepare (default: AgeDB)")
    parser.add_argument("--size", type=int, default=224, help="Size to which images will be resized (default: 48)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory containing AgeDB (default: ../Data/Age)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Path to the output directory for processed CSVs (default: ./data)")
    parser.add_argument("--dry_run", action="store_true", help="If set, will not write files but will log actions (default: False)")
    parser.add_argument("--children", action="store_true", help="If set, will only process images of children (ages 0-16) (default: False)")
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    DRY_RUN = args.dry_run
    size = args.size

    logs = []
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset.lower() == "all":
        for name in DATASET_CONFIG.keys():
            prepare_dataset(name, size, args.children, DATA_DIR, OUTPUT_DIR, dry_run=DRY_RUN, logs=logs)
    else:
        prepare_dataset(args.dataset, size, args.children, DATA_DIR, OUTPUT_DIR, dry_run=DRY_RUN, logs=logs)