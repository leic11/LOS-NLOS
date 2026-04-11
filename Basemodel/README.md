# Basemodel

Refactored project layout for the original single-file GNSS LOS/NLOS baseline code.

## Structure

```text
Basemodel/
  README.md
  requirements.txt
  .gitignore
  scripts/
    train.py
  src/
    basemodel/
      __init__.py
      config.py
      io_utils.py
      data.py
      model.py
      engine.py
      plotting.py
      train.py
```

## What This Contains

- Excel-based dataset loading and feature engineering
- Joint temporal-spatial LOS/NLOS model
- Training and evaluation loop
- Logging to both console and file
- Loss / F1 / Accuracy curve export

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python scripts/train.py
```

## Notes

- Update dataset paths in `src/basemodel/config.py` before running.
- The project keeps the original training logic, while splitting responsibilities into separate files.
