# Sepsis TD-GRU Pipeline 
## Project layout

```
sepsis_pipeline/
├── configs/
│   └── config.yaml                 # central knobs: paths, hyperparameters, seeds
├── src/
│   ├── config.py                   # config loader + reproducibility helpers
│   ├── data.py                     # tensorization with fixed Δt boundary
│   ├── losses.py                   # ATP loss + Weighted BCE
│   ├── models.py                   # TD-GRU, faithful GRU-D, baselines (PyTorch)
│   ├── training.py                 # multi-seed training harness
│   └── evaluation.py               # bootstrap, calibration, DCA, PhysioNet utility
├── notebooks/
│   ├── build_notebooks.py          # converts .py → .ipynb (run once)
│   ├── notebook_0_eda.py           # cohort attrition + Table 1 + missingness
│   ├── notebook_1_tensorize.py     # build X/M/D/y tensors
│   ├── notebook_2_baselines.py     # XGBoost + GRU + GRU-D + 1D-CNN
│   ├── notebook_3_transformer.py   # Transformer with continuous-time PE
│   ├── notebook_4_tdgru.py         # lead model + ablations + sensitivity sweep
│   ├── notebook_5_synthesis.py     # final tables + figures + statistical tests
│   └── notebook_6_temporal_split.py  # temporal train/val/test split + year distribution
├── tests/                          # pytest suite (43 tests)
├── requirements.txt                # pinned deps
└── README.md
```

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Edit `configs/config.yaml`

Set `paths.data_root` to the directory holding `final_cohort_3h - Work.csv.gz`
or the already-filtered `final_cohort_3h_filtered.parquet`. Everything else
defaults to sensible values.

### 3. Run tests

```bash
pytest tests/ -q
```

Expected: 43 passed.

### 4. Convert notebooks (once)

```bash
cd notebooks && python build_notebooks.py
```

This generates `notebook_*.ipynb` from the `.py` files. The `.py` files are
the source of truth (cleaner diffs in version control); the `.ipynb` files
are what you open in Jupyter / Colab.



## Reproducibility notes

- All neural runs use seeds from `cfg['seeds']`. With identical hardware and
  driver versions, results are bitwise reproducible. Cross-hardware
  reproducibility holds for metrics to ~3-4 decimal places.
- The bootstrap RNG is seeded separately (`cfg.bootstrap.random_seed`) so
  statistical tests are reproducible regardless of training-time seed.
- `tests/conftest.py` adds the project root to `sys.path` so notebooks
  importing `from src...` work without an editable install.

## Citation

If you use this code, cite the manuscript and this repository.

## License

MIT (recommended; check journal data-use agreements before redistributing).
