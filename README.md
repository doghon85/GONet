# GONet

PyTorch implementation of GONet for genomic prediction experiments on pig genotype and phenotype data.

## Files

- `main.py`: Runs 5-fold training and testing for `PFAI` and `MS`.
- `dataset.py`: Loads genotype, phenotype, train split, and test split files.
- `model.py`: Defines the GONet neural network.
- `train_test.py`: Contains training and evaluation loops.
- `utils.py`: Contains genomic relationship matrix, mixed model, and metric utilities.

## Data

The current script expects the following data layout relative to this directory:

```text
../../data/geno.txt
../../data/pheno.txt
../../data/5fold/train1.txt
../../data/5fold/test1.txt
...
../../data/5fold/train5.txt
../../data/5fold/test5.txt
```

The raw data files are not included in this repository.

## Environment

Required Python packages:

```text
torch
numpy
tqdm
```

## Run

```bash
python main.py
```

Training outputs are written to directories named like:

```text
result_1_PFAI_t0.1_lr0.0001/
result_1_MS_t0_lr0.0001/
```

These output directories are ignored by Git.
