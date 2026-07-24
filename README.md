# Age Estimation

This project includes data preparation, model training, evaluation and plotting tools to train models to estimate age from face pictures.
The datasets used for training are UTK-Face, AgeDB and IMDB-Wiki.

## Quick Start

1.Install dependencies

```bash
python -m pip install -r requirements.txt
```

2.Prepare the auxiliary derived CSVs

Convert the Datasets as saved from the links in the "Datasets" section to CSVs:

```bash
python -m src.data.prepare.datasets --data_dir /Path/To/Data
```

Options:

```bash
  -h, --help            show this help message and exit
  --dataset {all,AgeDB,UTKFace,Wiki,IMDb}
                        Dataset to prepare (default: AgeDB)
  --size SIZE           Size to which images will be resized (default: 48)
  --data_dir DATA_DIR   Path to the data directory containing AgeDB (default: ../Data/Age)
  --output_dir OUTPUT_DIR
                        Path to the output directory for processed CSVs (default: ./data)
  --dry_run             If set, will not write files but will log actions (default: False)
  --children            If set, will only process images of children (ages 0-16) (default: False)
```

3.Train a single model:

```bash
python -m src.training.train
```

Options:

```bash
  -h, --help            show this help message and exit
  --seed SEED           Random seed for reproducibility
  --dataset {agedb224,agedb48,utkface224,utkface48,imdb224,imdb48,wiki224,wiki48}
                        Dataset name (default: agedb)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs to wait for improvement before early stopping
  --output_dir OUTPUT_DIR
                        Directory to save checkpoints and logs
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --epochs EPOCHS       Number of epochs to train (default: 100)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.001)
  --model {vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet}
                        Model architecture to use (default: vgg16)
  --pretrained          Use pretrained weights for the model
  --freezed             Freeze the convolutional layers of the model
  --num_outputs NUM_OUTPUTS
                        Number of outputs for the regression head (default: 1)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the regression head (default: 0.5)
  --optimizer {adam,sgd,adamw}
                        Optimizer to use (default: adam)
  --cuda                Use CUDA for training if available
  --grad_clip GRAD_CLIP
                        Gradient clipping value (default: 0.0, no clipping)
  --data_augmentation_param DATA_AUGMENTATION_PARAM
                        Data augmentation parameter for rotation (degrees), scaling (percent), and shifting (percent) (default: 5)
  --resume              Resume training from the last checkpoint if available
  --no_checkpoint       Disable checkpoint saving
  --no_model_save       Disable model saving
  --lr_factor LR_FACTOR
                        Factor by which to reduce learning rate (default: 0.1)
  --lr_patience LR_PATIENCE
                        Number of epochs to wait for improvement before reducing learning rate (default: 10)
  --lr_threshold LR_THRESHOLD
                        Minimum change in loss to qualify as improvement (default: 1e-4)
  --lr_threshold_mode LR_THRESHOLD_MODE
                        Mode to use for determining if loss has improved (default: rel)
  --lr_cooldown LR_COOLDOWN
                        Number of epochs to wait before resuming normal operation after reducing learning rate (default: 0)
  --lr_min LR_MIN       Minimum learning rate (default: 0.0)
```

4.Optimize hyperparameters with Optuna

```bash
python src.training.run_optuna
```

Options:

```bash
  -h, --help            show this help message and exit
  --n_trials N_TRIALS   Number of trials for the optimization.
  --resume              Resume the optimization from the last state.
  --seed SEED           Random seed for reproducibility.
  --output_dir OUTPUT_DIR
                        Directory to save the Optuna study and logs.
  --log_name LOG_NAME   Name of the CSV log file for Optuna trials.
  --models MODELS       Comma-separated list of models to include in the optimization.
  --datasets DATASETS   Comma-separated list of datasets to include in the optimization.
  --learning_rate_range LEARNING_RATE_RANGE
                        Learning rate range for optimization (min,max).
  --batch_size_range BATCH_SIZE_RANGE
                        Batch size range for optimization (min,max,step).
  --dropout_rate_range DROPOUT_RATE_RANGE
                        Dropout rate range for optimization (min,max).
  --data_augmentation_range DATA_AUGMENTATION_RANGE
                        Data augmentation parameter range for optimization (min,max).
  --lr_factor_range LR_FACTOR_RANGE
                        Learning rate factor range for optimization (min,max).
  --lr_patience_range LR_PATIENCE_RANGE
                        Learning rate patience range for optimization (min,max).
  --lr_threshold_range LR_THRESHOLD_RANGE
                        Learning rate threshold range for optimization (min,max).
  --optimizers OPTIMIZERS
                        Comma-separated list of optimizers to include in the optimization.
  --epochs EPOCHS       Number of epochs for training in each trial.
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs to wait for improvement before early stopping.
  --lr_cooldown LR_COOLDOWN
                        Number of epochs to wait before resuming normal operation after lr has been reduced.
  --lr_min LR_MIN       Minimum learning rate after reduction.
  --lr_threshold_mode {rel,abs}
                        Mode for learning rate threshold ('rel' or 'abs').
  --pretrained          Use pretrained weights for the models.
  --freezed             Freeze the backbone of the model during training.
```

## Repository Layout

- `data/`: untracked, store the dataset csvs generated with /src/data/prepare_datasets.py here
- `models/`: model architectures for age estimation
- `output/`: training logs and model weights
- `src/data/`: data preparation and preprocessing
- `src/evaluation/`: evaluation script
- `src/plot/`: plotting script
- `src/training/`: training scripts
- `src/transforms/`: transforms methods for data loading and data augmentation
- `src/utils/`: shared dataset and utility modules

## Datasets

### AgeDB

- Source: Moschoglou, S., Papaioannou, A., Sagonas, C., Deng, J., Kotsia, I., & Zafeirou, S. (2017). Agedb: the first manually collected, in-the-wild age database, In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop*(pp. 51-59)
- Available at: [https://ibug.doc.ic.ac.uk/resources/agedb/](https://ibug.doc.ic.ac.uk/resources/agedb/)
- Licence: Available for non-commercial research purposes only

### UTKFace

- Source: Zhifei, Z., Yang, S., & Qi, H. (2017). Age Progression/Regression by Conditional Adversarial Autoencoder. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp.5810-5818)
- Available at: [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)
- Licence: Available for non-commercial research purposes only

### IMDb-Wiki

- Source: Rothe, R., Timofte, R., & Van Gool, L. (2018). Deep expectation of real and apparent age from a single image without facial landmarks. *International Journal of Computer Vision, 126(2)*, 144-157.
- Available at: [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- Licence: Available for academic research purposes only
