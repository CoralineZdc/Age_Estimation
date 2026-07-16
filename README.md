# Age Estimation

This project includes data preparation, model training, evaluation and plotting tools to train models to estimate age from face pictures.
The datasets used for training are UTK-Face, AgeDB and IMDB-Wiki.

## Quick Start

*In progress*

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

- AgeDB
Source: Moschoglou, S., Papaioannou, A., Sagonas, C., Deng, J., Kotsia, I., & Zafeirou, S. (2017). Agedb: the first manually collected, in-the-wild age database, In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop*(pp. 51-59)
Available at: [https://ibug.doc.ic.ac.uk/resources/agedb/](https://ibug.doc.ic.ac.uk/resources/agedb/)
Licence: Available for non-commercial research purposes only

- UTKFace 
Source: Zhifei, Z., Yang, S., & Qi, H. (2017). Age Progression/Regression by Conditional Adversarial Autoencoder. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp.5810-5818)
Available at: [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)
Licence: Available for non-commercial research purposes only

- IMDb-Wiki
Source: Rothe, R., Timofte, R., & Van Gool, L. (2018). Deep expectation of real and apparent age from a single image without facial landmarks. *International Journal of Computer Vision, 126(2)*, 144-157.
Available at: [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
Licence: Available for academic research purposes only