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
poetry run black machine_learning pos_tagging tests
poetry run isort machine_learning pos_tagging tests
```