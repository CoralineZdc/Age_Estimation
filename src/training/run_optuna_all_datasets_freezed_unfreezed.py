import sys
import os
import argparse
from types import SimpleNamespace
import subprocess


# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parsing_utils import *


def build_parser():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization on all datasets.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Path to the Python interpreter to use for running the training script.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for Optuna optimization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--datasets", type=parse_csv_strings, default="agedb,utkface,wiki,imdb", help="Comma-separated list of datasets to run Optuna on.")
    parser.add_argument("--training_mode", type=parse_csv_bool_tuples, default="(0,0);(1,1);(0,1)", help="semi-colon-separated list of training modes (tuples of boolean values) to run Optuna on.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(repo_root(), "output"), help="Directory to save Optuna results.")   
    parser.add_argument("--models", type=str, default="vgg16,vgg19,resnet18,efficientnet,mobilenet", help="Comma-separated list of models to include in the optimization.")
    parser.add_argument("--learning_rate_range", type=str, default="1e-5,1e-2", help="Learning rate range for optimization (min,max).")
    parser.add_argument("--batch_size_range", type=str, default="8,64,8", help="Batch size range for optimization (min,max,step).")
    parser.add_argument("--dropout_rate_range", type=str, default="0.1,0.5", help="Dropout rate range for optimization (min,max).")
    parser.add_argument("--data_augmentation_range", type=str, default="0,20", help="Data augmentation parameter range for optimization (min,max).")
    parser.add_argument("--lr_factor_range", type=str, default="0.1,0.5", help="Learning rate factor range for optimization (min,max).")
    parser.add_argument("--lr_patience_range", type=str, default="5,20", help="Learning rate patience range for optimization (min,max).")
    parser.add_argument("--lr_threshold_range", type=str, default="1e-5,1e-3", help="Learning rate threshold range for optimization (min,max).")
    parser.add_argument("--optimizers", type=str, default="adam,sgd,adamw", help="Comma-separated list of optimizers to include in the optimization.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training in each trial.")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--lr_cooldown", type=int, default=0, help="Number of epochs to wait before resuming normal operation after lr has been reduced.")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate after reduction.")
    parser.add_argument("--lr_threshold_mode", type=str, default="rel", choices=["rel", "abs"], help="Mode for learning rate threshold ('rel' or 'abs').")
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_parser()

    dataset_list = args.datasets
    print(f"Datasets to run Optuna on: {dataset_list}")
    training_mode = args.training_mode

    explored_trials = []
    if args.resume:
        logs = [file for file in os.listdir(os.path.join(args.output_dir, "optuna")) if file.endswith(".csv")]
        for log in logs:
            parts = log.split("_")
            parts[-1] = parts[-1].replace(".csv", "")
            dataset = parts[1]
            mode = parts[2]
            if dataset in dataset_list and mode in training_mode:
                explored_trials.append((dataset, mode))

    print(f"Explored trials: {explored_trials}")
    for dataset in dataset_list:
        for mode in training_mode:
            pretrained, freezed = mode
            pretrained_str = "pretrained" if pretrained else "scratch"
            freezed_str = "freezed" if freezed else "unfreezed"
            if (dataset, mode) in explored_trials:
                print(f"Skipping dataset: {dataset}, training mode: {pretrained_str}_{freezed_str} (already explored)")
                continue
            print(f"Running Optuna on dataset: {dataset}, training mode: {pretrained_str}_{freezed_str}")
            

            command = [
                args.python,
                os.path.join(repo_root(), "src", "training", "run_optuna.py"),
                "--n_trials", str(args.n_trials),
                "--seed", str(args.seed),
                "--output_dir", args.output_dir,
                "--log_name", f"log_{dataset}_{pretrained_str}_{freezed_str}.csv",
                "--models", args.models,
                "--datasets", f"{dataset}48,{dataset}224",
                "--learning_rate_range", args.learning_rate_range,
                "--batch_size_range", args.batch_size_range,
                "--dropout_rate_range", args.dropout_rate_range,
                "--data_augmentation_range", args.data_augmentation_range,
                "--lr_factor_range", args.lr_factor_range,
                "--lr_patience_range", args.lr_patience_range,
                "--lr_threshold_range", args.lr_threshold_range,
                "--optimizers", args.optimizers,
                "--epochs", str(args.epochs),
                "--early_stopping_patience", str(args.early_stopping_patience),
                "--lr_cooldown", str(args.lr_cooldown),
                "--lr_min", str(args.lr_min),
                "--lr_threshold_mode", args.lr_threshold_mode
            ]
            if freezed:
                command.append("--freezed")
            if pretrained:
                command.append("--pretrained")

            subprocess.run(command)
    
