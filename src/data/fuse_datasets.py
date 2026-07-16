import os
from pdb import main
import sys
import argparse
import pandas as pd

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path and os.path.exists(os.path.join(project_root, "Age_Estimation")):
    sys.path.insert(0, project_root)
project_root = os.path.join(project_root, "Age_Estimation")

from Age_Estimation.src.utils.parsing_utils import *


def get_dataset_paths(data_dir, dataset_names, image_size, population, split="train"):
    dataset_paths = []
    print(f"Searching for datasets in '{data_dir}' for split '{split}' with population '{population}' and image size '{image_size}':", end="\r")
    for dataset_name in dataset_names:
        file_name = f"{split}-{'children-' if population == 'children' else ''}{dataset_name}{image_size}.csv"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"    - Found dataset: {dataset_name} at {file_path:<20}", end="\r")
            dataset_paths.append((file_path, dataset_name))
        else:
            print(f"Warning: The dataset file '{file_path}' does not exist. Skipping this dataset.{'':<20}", end="\r")
    return dataset_paths


def fuse_datasets(dataset_paths, dataset_names, dataset_proportion, image_size, population, split, seed, dry_run=False):
    fused_data = pd.DataFrame()
    for dataset_path, dataset_name in dataset_paths:
        print(f"Fusing dataset: {dataset_name} from {dataset_path} with proportion {dataset_proportion*100}%{'':<30}", end="\r")

        data = pd.read_csv(dataset_path)
        data = data.sample(frac=dataset_proportion, random_state=seed)
        data['dataset_name'] = dataset_name
        fused_data = pd.concat([fused_data, data], ignore_index=True)

    fused_data = fused_data.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the fused dataset

    dataset_names.sort(key=lambda x: x.lower())
    dataset_names_str = "-".join(dataset_names)
    fused_file_name = f"{split}-{'children-' if population == 'children' else ''}fused{dataset_proportion*100:.0f}-{dataset_names_str}{image_size}.csv"
    out_path = os.path.join(project_root, "data", fused_file_name)

    if not dry_run:
        fused_data.to_csv(out_path, index=False)
        print(f"Fused dataset saved to {out_path}")
    else:
        print(f"Dry run: Fused dataset would be saved to {out_path}")
    print(f"Fused dataset contains {len(fused_data)} samples from datasets: {', '.join(dataset_names)}")
    return fused_data


def main():
    args = build_parser()
    data_dir = args.data_dir
    dataset_names = args.dataset_names
    dataset_proportion = args.dataset_proportion
    image_size = args.image_size
    population = args.population
    seed = args.seed
    dry_run = args.dry_run
    splits = ["train", "val", "test"]

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: The specified data directory '{data_dir}' does not exist.")
        sys.exit(1)

    for split in splits:
        print(f"\nProcessing split: {split}")
        dataset_paths = get_dataset_paths(data_dir, dataset_names, image_size, population, split)
        fuse_datasets(dataset_paths, dataset_names, dataset_proportion, image_size, population, split, seed, dry_run=dry_run)


def build_parser():
    parser = argparse.ArgumentParser(description="Fuse datasets for age estimation.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the directory containing the datasets.")
    parser.add_argument("--dataset_names", type=parse_csv_strings, default="utkface, agedb, imdb, wiki", help="Comma-separated list of dataset names to fuse.")
    parser.add_argument("--dataset_proportion", type=float, default=1, help="Proportion of samples to include in the fused dataset.")
    parser.add_argument("--image_size", type=int, default=48, help="Size of the images in the fused dataset (width,height).")
    parser.add_argument("--population", type=str, default="all", choices=["all", "children"], help="Population to include in the fused dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dry_run", action="store_true", help="If set, the script will not save the fused dataset to disk.")
    return parser.parse_args()



if __name__ == "__main__":
    main()