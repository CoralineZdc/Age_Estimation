

import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parsing_utils import repo_root
    

def plot_loss_curve(csv_path: Path, output_dir: Path) -> Path:

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if the required columns exist
    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'train_loss' and 'val_loss' columns.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    simu_params = os.path.dirname(csv_path).split('/')[-1]
    simu_params_list = simu_params.split('_')
    for simu_param_id in range(len(simu_params_list)):
        simu_param = simu_params_list[simu_param_id]
        for i in range(len(simu_param)):
            if simu_param[i].isdigit():
                simu_params_list[simu_param_id] = simu_param[:i] + '=' + simu_param[i:]
                break
    if len(simu_params_list) % 6 != 0:
        for _ in range(6 - len(simu_params_list) % 6):
            simu_params_list.append('')
    simu_params_str = ''
    for i in range(len(simu_params_list) // 6):
        simu_params_str += ', '.join(simu_params_list[6*i:6*i+6]).strip(', ') + '\n'
    simu_params_str = simu_params_str.strip()  # Remove the trailing newline
    print(f"Simu params: {simu_params_str}")

    # Plotting the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='o')
    plt.title(f"{simu_params_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Save the plot to the output directory
    output_file = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(output_file)
    plt.close()

    return output_file


def discover_csvs(dir: Path) -> list[Path]:
    return sorted(
        path for path in Path(dir).rglob("*.csv")
        if path.is_file() and path.name.lower().endswith(".csv")
    )


def run(args: argparse.Namespace) -> tuple[int, list[str]]:
    dir = Path(args.dir)
    repo_root_path = Path(repo_root())

    if not os.path.isabs(dir):
        dir = os.path.join(repo_root_path, dir)

    csv_paths = discover_csvs(dir)

    if not csv_paths:
        raise SystemExit(f"No CSV files found in {dir}")
    
    saved = 0
    skipped: list[str] = []

    for csv_path in csv_paths:
        try:
            out_path = plot_loss_curve(csv_path, dir)
            print(f"Saved {out_path}")
            saved += 1
        except Exception as exc:  
            skipped.append(f"{csv_path}: {exc}")

    return saved, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw Valence, Arousal, and Dominance histograms for VAD CSV datasets.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing log csv files and where the loss curve images will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    saved, skipped = run(args)

    print(f"Plotted {saved} dataset(s).")
    if skipped:
        print("Skipped datasets:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()