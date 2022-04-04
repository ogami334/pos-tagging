# Train Baseline Model
```bash
poetry run train.py configs/baseline.json results/baseline
```

# Code Formatting
```bash
poetry run black machine_learning pos_tagging tests train.py make_submission_file.py
poetry run isort machine_learning pos_tagging tests train.py make_submission_file.py
```