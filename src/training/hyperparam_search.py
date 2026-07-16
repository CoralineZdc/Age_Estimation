import argparse
import csv
import os
import sys
from datetime import datetime
import itertools
from pathlib import Path
import time
import subprocess


def parse_csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_best_metrics(log_path):
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None, None, None, None

    best_mae = None
    best_epoch = None
    last_mae = None
    last_epoch = None

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                mae = float(row["val_loss"])
            except Exception:
                continue

            last_mae = mae
            last_epoch = epoch
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_epoch = epoch

    return best_mae, best_epoch, last_mae, last_epoch




def main():
    parser = argparse.ArgumentParser(description="Grid search for hyperparameter tuning.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Path to the Python interpreter to use for running the training script.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--datasets", type=str, default="agedb48", help="List of datasets to train on.")
    parser.add_argument("--early_stopping_patience", type=int, default=40, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--batch_sizes", type=str, default="32", help="List of batch sizes to try.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for each configuration.")
    parser.add_argument("--learning_rates", type=str, default="0.001", help="List of learning rates to try.")
    parser.add_argument("--models", type=str, default="efficientnet", help="List of models to train.")
    parser.add_argument("--dropout_rates", type=str, default="0.3", help="List of dropout rates to try.")
    parser.add_argument("--optimizers", type=str, default="adamw", help="List of optimizers to try.")
    parser.add_argument("--data_augmentation_params", type=str, default="5", help="List of data augmentation parameters to try.")
    parser.add_argument("--lr_factors", type=str, default="0.1,", help="List of learning rate factors to try.")
    parser.add_argument("--lr_patiences", type=str, default="10", help="List of learning rate patience values to try.")
    parser.add_argument("--lr_thresholds", type=str, default="0.0001", help="List of learning rate thresholds to try.")
    parser.add_argument("--lr_threshold_mode", type=str, default="rel", help="Mode to use for determining if loss has improved (default: rel).")
    parser.add_argument("--lr_cooldowns", type=str, default="0", help="List of learning rate cooldown values to try.")
    parser.add_argument("--lr_mins", type=str, default="0", help="List of learning rate minimum values to try.")
    parser.add_argument("--delete_models", action="store_true", help="Delete models after training to save space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to perform computations on (default: cuda).")
    opt = parser.parse_args()

    datasets = parse_csv_strings(opt.datasets)
    batch_sizes = parse_csv_ints(opt.batch_sizes)
    learning_rates = parse_csv_floats(opt.learning_rates)
    models = parse_csv_strings(opt.models)
    dropout_rates = parse_csv_floats(opt.dropout_rates)
    optimizers = parse_csv_strings(opt.optimizers)
    data_augmentation_params = parse_csv_floats(opt.data_augmentation_params)
    lr_factors = parse_csv_floats(opt.lr_factors)
    lr_patiences = parse_csv_ints(opt.lr_patiences)
    lr_thresholds = parse_csv_floats(opt.lr_thresholds)
    lr_threshold_mode = opt.lr_threshold_mode
    lr_cooldowns = parse_csv_ints(opt.lr_cooldowns)
    lr_mins = parse_csv_floats(opt.lr_mins)

    bs_max = {
        "mobilefacenet":8, 
        "efficientnet": 72,
        "vgg16": 54,
        "vgg19": 48,
        "resnet18": 28
    }

    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    stamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    output_dir = root / "output" 
    summary_dir = output_dir / "gridsearch"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "grid-{}.csv".format(stamp)
    
    summary_fields = [
        "dataset",
        "model",
        "batch_size",
        "learning_rate",
        "dropout_rate",
        "data_augmentation",
        "lr_factor",
        "lr_patience",
        "lr_threshold",
        "lr_threshold_mode",
        "lr_cooldown",
        "lr_min",
        "duration",
        "best_val_mae",
        "best_epoch",
        "final_val_mae",
        "final_test_mae",
        "final_epoch",
        "error"
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        writer.writeheader()
    """
    grid = list(itertools.product(
        datasets,
        models,
        batch_sizes,
        learning_rates,
        dropout_rates,
        optimizers,
        data_augmentation_params,
        lr_factors,
        lr_patiences,
        lr_thresholds,
        lr_cooldowns,
        lr_mins,
    ))

    print("Total runs:", len(grid))

    run_counter = 0
    """

    for model in models:
        run_counter = 0
        max_bs = bs_max.get(model, 100)

        valid_batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]
        if max_bs not in batch_sizes:
            valid_batch_sizes.append(max_bs)
        
        current_grid = list(itertools.product(
            datasets,
            [model],
            valid_batch_sizes,
            learning_rates,
            dropout_rates,
            optimizers,
            data_augmentation_params,
            lr_factors,
            lr_patiences,
            lr_thresholds,
            lr_cooldowns,
            lr_mins,
        ))

        print(f"Total runs for model {model}: {len(current_grid)}                                                                                                                                                                                   ")

        for run_item in current_grid:
            run_counter +=1
            dataset, _, batch_size, learning_rate, dropout_rate, optimizer, data_augmentation_param, lr_factor, lr_patience, lr_threshold, lr_cooldown, lr_min = run_item

            print(f"|{'█'*int((run_counter) / len(current_grid) *40)}{' '*int(40 - int((run_counter) / len(current_grid) * 40))}| {(run_counter) / len(current_grid) * 100 :.2f}% [{run_counter}/{len(current_grid)}]", end="\r")

            data_augmentation = 1 if data_augmentation_param > 1e-4 else 0
            rotation = data_augmentation_param if data_augmentation else 0.0
            scale = data_augmentation_param*1e-2 if data_augmentation else 0.0
            hor_shift = data_augmentation_param*1e-2 if data_augmentation else 0.0
            ver_shift = data_augmentation_param*1e-2 if data_augmentation else 0.0


            command = [
                opt.python,
                str(root / "src" / "training" / "train.py"),
                "--seed", str(opt.seed),
                "--dataset", dataset,
                "--early_stopping_patience", str(opt.early_stopping_patience),
                "--epochs", str(opt.epochs),
                "--model", model,
                "--batch_size", str(batch_size),
                "--learning_rate", str(learning_rate),
                "--dropout_rate", str(dropout_rate),
                "--optimizer", optimizer,
                "--data_augmentation", str(data_augmentation),
                "--scale", str(scale),
                "--rotation", str(rotation),
                "--hor_shift", str(hor_shift),
                "--ver_shift", str(ver_shift),
                "--lr_factor", str(lr_factor),
                "--lr_patience", str(lr_patience),
                "--lr_threshold", str(lr_threshold),
                "--lr_threshold_mode", lr_threshold_mode,
                "--lr_cooldown", str(lr_cooldown),
                "--lr_min", str(lr_min),
                "--early_stopping_patience", str(opt.early_stopping_patience),
                "--no_checkpoint" # Disable checkpoint saving during grid search
            ]

            start = time.time()
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )

            end = time.time()
            duration = end - start
            duration = time.strftime("%H:%M:%S", time.gmtime(duration))
            error = None if result.returncode == 0 else result.stderr.strip()

            # Extract metrics from the output
            best_val_mae = None
            best_epoch = None
            final_val_mae = None
            final_test_mae = None
            final_epoch = None

            folder_name = f"{model}/seed{opt.seed}_dataset-{dataset}_opt-{optimizer}_lr{learning_rate:.5f}_bs{batch_size}_dropout{dropout_rate:.1f}{f'_scale{scale:.1f}_rot{rotation:.1f}_hor-shift{hor_shift:.1f}_ver-shift{ver_shift:.1f}' if data_augmentation else ''}_lr-factor{lr_factor:.1f}_lr-patience{lr_patience}_lr-threshold{lr_threshold:.4f}_lr-threshold-mode-{lr_threshold_mode}_lr-cooldown{lr_cooldown}_lr-min{lr_min:.1f}"
            path = output_dir / folder_name
            log_path = path / "log.csv"
            if log_path.exists():
                best_val_mae, best_epoch, final_val_mae, final_epoch = parse_best_metrics(log_path)

            state_dict_path = path / "best_model_state.pth"
            test_command = [
                opt.python,
                str(root / "src" / "evaluation" / "evaluation.py"),
                "--model", model,
                "--dataset", dataset,
                "--state-dict-path", str(state_dict_path),
                "--device", opt.device,
                "--batch-size", str(batch_size)
            ]
            test_result = subprocess.run(test_command, capture_output=True, text=True)
            if test_result.returncode == 0:
                final_test_mae = test_result.stdout.strip().split("Mean Absolute Error:")[-1].strip()
            else:
                print("Error during evaluation:", test_result.stderr)
                final_test_mae = None

            plot_command = [
                opt.python,
                str(root / "src" / "plot" / "loss_curve.py"),
                "--dir", str(path),
            ]
            subprocess.run(plot_command, capture_output=True, text=True)

            with summary_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
                writer.writerow({
                    "dataset": dataset,
                    "model": model,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "data_augmentation": data_augmentation,
                    "lr_factor": lr_factor,
                    "lr_patience": lr_patience,
                    "lr_threshold": lr_threshold,
                    "lr_threshold_mode": lr_threshold_mode,
                    "lr_cooldown": lr_cooldown,
                    "lr_min": lr_min,
                    "duration": duration,
                    "best_val_mae": best_val_mae,
                    "best_epoch": best_epoch,
                    "final_val_mae": final_val_mae,
                    "final_test_mae": final_test_mae,
                    "final_epoch": final_epoch,
                    "error": error
            })
            
            if opt.delete_models:
                if path.exists():
                    for file in path.glob("*"):
                        file.unlink()
                    path.rmdir()

    """
    for dataset, model, batch_size, learning_rate, dropout_rate, optimizer, data_augmentation_param, lr_factor, lr_patience, lr_threshold, lr_cooldown, lr_min in grid:
        run_counter += 1

        print(f"|{'█'*int((run_counter) / len(grid) *40)}{' '*int(40 - int((run_counter) / len(grid) * 40))}| {(run_counter) / len(grid) * 100 :.2f}% [{run_counter}/{len(grid)}]", end="\r")

        if batch_size > bs_max.get(model, 100):
            continue

        data_augmentation = 1 if data_augmentation_param > 1e-4 else 0
        rotation = data_augmentation_param if data_augmentation else 0.0
        scale = data_augmentation_param*1e-2 if data_augmentation else 0.0
        hor_shift = data_augmentation_param*1e-2 if data_augmentation else 0.0
        ver_shift = data_augmentation_param*1e-2 if data_augmentation else 0.0


        command = [
            opt.python,
            str(root / "src" / "training" / "train.py"),
            "--seed", str(opt.seed),
            "--dataset", dataset,
            "--early_stopping_patience", str(opt.early_stopping_patience),
            "--epochs", str(opt.epochs),
            "--model", model,
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--dropout_rate", str(dropout_rate),
            "--optimizer", optimizer,
            "--data_augmentation", str(data_augmentation),
            "--scale", str(scale),
            "--rotation", str(rotation),
            "--hor_shift", str(hor_shift),
            "--ver_shift", str(ver_shift),
            "--lr_factor", str(lr_factor),
            "--lr_patience", str(lr_patience),
            "--lr_threshold", str(lr_threshold),
            "--lr_threshold_mode", lr_threshold_mode,
            "--lr_cooldown", str(lr_cooldown),
            "--lr_min", str(lr_min),
            "--early_stopping_patience", str(opt.early_stopping_patience),
        ]

        start = time.time()
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        end = time.time()
        duration = end - start
        error = None if result.returncode == 0 else result.stderr.strip()

        # Extract metrics from the output
        best_val_mae = None
        best_epoch = None
        final_val_mae = None
        final_test_mae = None
        final_epoch = None

        folder_name = f"{model}/seed{opt.seed}_dataset-{dataset}_opt-{optimizer}_lr{learning_rate:.5f}_bs{batch_size}_dropout{dropout_rate:.1f}{f'_data-aug_scale{scale:.1f}_rot{rotation:.1f}_hor-shift{hor_shift:.1f}_ver-shift{ver_shift:.1f}' if data_augmentation else ''}_lr-factor{lr_factor:.1f}_lr-patience{lr_patience}_lr-threshold{lr_threshold:.4f}_lr-threshold-mode-{lr_threshold_mode}_lr-cooldown{lr_cooldown}_lr-min{lr_min:.1f}"
        path = output_dir / folder_name
        log_path = path / "log.csv"
        if log_path.exists():
            best_val_mae, best_epoch, final_val_mae, final_epoch = parse_best_metrics(log_path)

        state_dict_path = path / "best_model_state.pth"
        test_command = [
            opt.python,
            str(root / "src" / "evaluation" / "evaluation.py"),
            "--model", model,
            "--dataset", dataset,
            "--state-dict-path", str(state_dict_path),
            "--device", opt.device,
            "--batch-size", str(batch_size)
        ]
        test_result = subprocess.run(test_command, capture_output=True, text=True)
        if test_result.returncode == 0:
            final_test_mae = test_result.stdout.strip().split("Mean Absolute Error:")[-1].strip()
        else:
            print("Error during evaluation:", test_result.stderr)
            final_test_mae = None

        with summary_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
            writer.writerow({
                "dataset": dataset,
                "model": model,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "data_augmentation": data_augmentation,
                "lr_factor": lr_factor,
                "lr_patience": lr_patience,
                "lr_threshold": lr_threshold,
                "lr_threshold_mode": lr_threshold_mode,
                "lr_cooldown": lr_cooldown,
                "lr_min": lr_min,
                "duration": duration,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "final_val_mae": final_val_mae,
                "final_test_mae": final_test_mae,
                "final_epoch": final_epoch,
                "error": error
            })
    """
    
    print(f"\nGrid search completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()

    