import sys
import optuna 
import os
import torch
import argparse
import shutil
import random
import numpy as np
from types import SimpleNamespace

from train import run_training

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.training_utils import set_seed
from src.utils.parsing_utils import *



def save_csv_callback(study, trial, log_name="log.csv"):
    # Save the study's trials to a CSV file after each trial
    output_dir = os.path.join(repo_root(), "output/optuna")
    os.makedirs(output_dir, exist_ok=True)

    if trial.state.is_finished():
        df = study.trials_dataframe(attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "state", "params", "user_attrs", "system_attrs"))
        csv_path = os.path.join(output_dir, log_name)
        df.to_csv(csv_path, index=False)
        print(f"Trial {len(study.trials)} saved to {csv_path}")


def objective(trial, args):

    models = args.models
    datasets = args.datasets
    learning_rate_range = args.learning_rate_range
    batch_size_range =  args.batch_size_range
    dropout_rate_range = args.dropout_rate_range
    data_augmentation_range = args.data_augmentation_range
    lr_factor_range = args.lr_factor_range
    lr_patience_range = args.lr_patience_range
    lr_threshold_range = args.lr_threshold_range
    optimizers = args.optimizers

    bs_max = {
        "mobilefacenet":8, 
        "efficientnet": 72,
        "vgg16": 54,
        "vgg19": 48,
        "resnet18": 28
    }

    dataset = trial.suggest_categorical("dataset", datasets)
    model_name = trial.suggest_categorical("model", models)
    learning_rate = trial.suggest_float("learning_rate", learning_rate_range[0], learning_rate_range[1], log=True)

    max_bs_for_model = bs_max.get(model_name, 32)
    batch_size = trial.suggest_int("batch_size", batch_size_range[0], min(max_bs_for_model, batch_size_range[1]), batch_size_range[2])  # Suggest batch sizes in multiples of 8 up to the max for the model
    dropout_rate = trial.suggest_float("dropout_rate", dropout_rate_range[0], dropout_rate_range[1])
    optimizer_name = trial.suggest_categorical("optimizer", optimizers)

    # Data augmentation parameters
    data_augmentation_param = trial.suggest_int("data_augmentation_param", data_augmentation_range[0], data_augmentation_range[1])

    # Scheduler parameters
    lr_factor = trial.suggest_float("lr_factor", lr_factor_range[0], lr_factor_range[1])
    lr_patience = trial.suggest_int("lr_patience", lr_patience_range[0], lr_patience_range[1])
    lr_threshold = trial.suggest_float("lr-threshold", lr_threshold_range[0], lr_threshold_range[1], log=True)

    opt = SimpleNamespace(
        dataset=dataset,
        model=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        optimizer=optimizer_name,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        lr_threshold=lr_threshold,
        data_augmentation_param=data_augmentation_param,
        seed=args.seed,
        epochs=args.epochs,
        num_outputs=1,
        grad_clip=0.0,
        cuda=torch.cuda.is_available(),
        early_stopping_patience=args.early_stopping_patience,
        lr_cooldown=args.lr_cooldown,
        lr_min=args.lr_min,
        lr_threshold_mode=args.lr_threshold_mode,
        no_checkpoint=True,
        resume=False,
        no_model_save=True,
        output_dir=os.path.join(repo_root(), "output"),
        pretrained=args.pretrained,
        freezed=args.freezed,
    )

    data_augmentation = opt.data_augmentation_param > 0
    scale = opt.data_augmentation_param / 100.0
    rotation = opt.data_augmentation_param
    hor_shift = opt.data_augmentation_param / 100.0
    ver_shift = opt.data_augmentation_param / 100.0

    folder_name = f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}{f'_scale{scale:.1f}_rot{rotation:.1f}_hor-shift{hor_shift:.1f}_ver-shift{ver_shift:.1f}' if data_augmentation else ''}_lr-factor{opt.lr_factor:.1f}_lr-patience{opt.lr_patience}_lr-threshold{opt.lr_threshold:.4f}_lr-threshold-mode-{opt.lr_threshold_mode}_lr-cooldown{opt.lr_cooldown}_lr-min{opt.lr_min:.1f}"
    folder_path = os.path.join(opt.output_dir, folder_name)

    # Run training and return the validation loss
    try:
        best_score = run_training(opt, trial=trial)
        return best_score
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        raise optuna.exceptions.TrialPruned()  # Prune the trial if an exception occurs
    finally:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"Failed to remove folder {folder_path}: {e}")
        else:
            print(f"Folder {folder_path} does not exist, skipping removal.")
    

def build_parser():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for age estimation.")
    parser.add_argument("--n_trials", type=int, default=150, help="Number of trials for the optimization.")
    parser.add_argument("--resume", action="store_true", help="Resume the optimization from the last state.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(repo_root(), "output"), help="Directory to save the Optuna study and logs.")
    parser.add_argument("--log_name", type=str, default="log.csv", help="Name of the CSV log file for Optuna trials.")
    parser.add_argument("--models", type=parse_csv_strings, default="vgg16,vgg19,resnet18,efficientnet,mobilenet", help="Comma-separated list of models to include in the optimization.")
    parser.add_argument("--datasets", type=parse_csv_strings, default="agedb48,agedb224,utkface48,utkface224,wiki48,wiki224,imdb48,imdb224", help="Comma-separated list of datasets to include in the optimization.")
    parser.add_argument("--learning_rate_range", type=parse_csv_floats, default="1e-5,1e-2", help="Learning rate range for optimization (min,max).")
    parser.add_argument("--batch_size_range", type=parse_csv_ints, default="8,64,8", help="Batch size range for optimization (min,max,step).")
    parser.add_argument("--dropout_rate_range", type=parse_csv_floats, default="0.1,0.5", help="Dropout rate range for optimization (min,max).")
    parser.add_argument("--data_augmentation_range", type=parse_csv_ints, default="0,20", help="Data augmentation parameter range for optimization (min,max).")
    parser.add_argument("--lr_factor_range", type=parse_csv_floats, default="0.1,0.5", help="Learning rate factor range for optimization (min,max).")
    parser.add_argument("--lr_patience_range", type=parse_csv_ints, default="5,20", help="Learning rate patience range for optimization (min,max).")
    parser.add_argument("--lr_threshold_range", type=parse_csv_floats, default="1e-5,1e-3", help="Learning rate threshold range for optimization (min,max).")
    parser.add_argument("--optimizers", type=parse_csv_strings, default="adam,sgd,adamw", help="Comma-separated list of optimizers to include in the optimization.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training in each trial.")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--lr_cooldown", type=int, default=0, help="Number of epochs to wait before resuming normal operation after lr has been reduced.")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate after reduction.")
    parser.add_argument("--lr_threshold_mode", type=str, default="rel", choices=["rel", "abs"], help="Mode for learning rate threshold ('rel' or 'abs').")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the models.")
    parser.add_argument("--freezed", action="store_true", help="Freeze the backbone of the model during training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_parser()

    set_seed(args.seed)

    root = repo_root()
    db_dir = os.path.join(args.output_dir, "optuna")
    db_path = os.path.join(db_dir, "optuna_study.db")

    storage_name=f"sqlite:///{db_path}"
    study_name = "age_estimation_optimization"

    os.makedirs(db_dir, exist_ok=True)

    load_if_exists = args.resume

    if not load_if_exists:
        if os.path.exists(db_path):
            print(f"Removing existing database at {db_path} for a fresh start.")
            os.remove(db_path)
        else:
            print(f"No existing database found at {db_path}. Starting a new study.")

    print(f"Using storage: {storage_name}")
    print(f"Study name: {study_name}")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5),
        load_if_exists=load_if_exists
    )

    if len(study.trials) > 0:
        print(f"Resuming study with {len(study.trials)} existing trials.")
        print(f"Best trial so far: Value: {study.best_trial.value}, Params: {study.best_trial.params}" )
    else:
        print("Starting a new study.")

    print(f"Saving logs to: {os.path.join(db_dir, args.log_name)}")
    try:
        study.optimize(
            lambda trial: objective(trial, args=args), 
            n_trials=args.n_trials, 
            n_jobs=1, 
            callbacks=[lambda study, trial: save_csv_callback(study, trial, log_name=args.log_name)]
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Optimization finished. Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

