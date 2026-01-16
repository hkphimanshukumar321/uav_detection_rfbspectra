# RF-CNN Backbone (Notebook Split for GitHub)

This repository contains a **GitHub-friendly, split** version of the original notebook `RF_CNN_Backbone (2).ipynb`.

## Structure

```
RF-CNN-Backbone_repo/
  notebooks/
    00_index.ipynb
    01_setup.ipynb
    02_data.ipynb
    03_model.ipynb
    04_train.ipynb
    05_evaluate.ipynb
  assets/
    (generated figures / model architecture image)
  models/
    (saved model checkpoints)
  src/
    (optional: move reusable code here)
  requirements.txt
  .gitignore
```

## Execution order

Open `notebooks/00_index.ipynb` and run notebooks in order:

1. `01_setup.ipynb`
2. `02_data.ipynb`
3. `03_model.ipynb`
4. `04_train.ipynb`
5. `05_evaluate.ipynb`

Each notebook contains **Prev / Index / Next** navigation links at the top.

## Important edits made for GitHub portability

- Removed `pip install ...` notebook cells (use `requirements.txt` instead).
- Replaced hard-coded absolute paths (e.g., `/home/...`) with project-relative paths:
  - Dataset: `data/Condata` (you provide locally; not committed)
  - Model architecture image: `assets/DenseNet_RF_CNN_Model_Architecture.png`
  - Saved model: `models/att_backbone.h5`

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Place your dataset locally at:

```
data/Condata/
  0/
  1/
  ...
  23/
```

`data/` is ignored by git by default.

## Notes (recommended next step)

For better reproducibility and code review, consider moving reusable code from notebooks into `src/` (e.g., `src/data.py`, `src/model.py`, `src/train.py`) and keeping notebooks as thin experiment drivers.
