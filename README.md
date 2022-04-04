# Part-of-Speech (PoS) Tagging

## Installation
We recommend using python of the version ^3.8.
Use the following command to install the dependencies.
```
poetry install
```

## Train a baseline model
```bash
# Usage: train.py CONFIG_PATH RESULT_SAVE_DIRECTORY
poetry run python train.py configs/baseline.json results/baseline
```

## Make a submission file
```bash
# Usage: train.py RESULT_SAVE_DIRECTORY
poetry run python make_submission_file.py results/baseline
```

## Code Formatting
```bash
poetry run black pos_tagging pos_tagging tests *.py
poetry run isort pos_tagging pos_tagging tests *.py
```