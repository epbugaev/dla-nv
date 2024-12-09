# DLA Homework NV, Bugaev Egor 213 group 

## About

This repository contains a solve DLA HW 3, it is based on the template for [PyTorch](https://pytorch.org/)-based Deep Learning projects. 

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).
    `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Inference on directory with custom audio files
Please make sure you did not switch inference and inference_text_custom configs
```
python3 synthesize --folder PATH_TO_YOUR_FOLDER
```

To change saving folder, `save_path` parameter in `inference.yaml`

## Inference on directory with custom text files
Please copy `inference_text_custom.yaml` config and paste to `inference.yaml` (while `inference.yaml` save somewhere else). Then run:
```
python3 synthesize --folder PATH_TO_YOUR_FOLDER
```

To change saving folder, `save_path` parameter in new `inference.yaml`

## Inference on custom text
Please make sure there is `inference_text_custom.yaml` config (here script uses such name).
```
python3 synthesize --text "YOUR TEXT"
```
To change saving folder, `save_path` parameter in `inference_text_custom.yaml`

## Credits

Huge thanks for the Deep Learning in Audio course in Higher School of Economics for this template and the homework that inspired this repo.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
