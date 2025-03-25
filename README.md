This repository contains the official implementation for the ``Text-driven Online Action Detection´´ (TOAD) method, published in [Integrated Computer-Aided Engineering](https://journals.sagepub.com/doi/full/10.1177/10692509241308069) ([arXiv](https://arxiv.org/abs/2501.13518)).

## Requirements & Setup
We provide a Dockerfile to create an reproducible environment using Docker and NVIDIA Docker, though Conda or Python environments may be used as well.
The Docker image is built locally using the provided Dockerfile with:

```
cd scripts
docker build -t toad .
```

## Feature Extraction
CLIP features (frame and text) are preprocessed and stored in .pt files for faster training. You may use the files in the clip directory to extract the features. The files will automatically download the corresponding weights from OpenAI API.

## Annotations
You may find the annotations we used processed in a pickle file in the annotations directory. You may use the PyTorch dataset class in the dataset directory as an example to open these files.

## Running 
The configuration files in the config folder provide the necessary parameters to run the code. The main.py file is the entry point to run the code.
Please use:
```bash
bash scripts/run_train.sh <config_file>
```

## Citation
If you find our paper useful, please consider citing our work.

## Contact
If you have any questions, please contact me at: ([mbenavent@dtic.ua.es](mailto:mbenavent@dtic.ua.es))