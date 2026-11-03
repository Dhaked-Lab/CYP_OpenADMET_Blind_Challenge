# Reported test environment (Windows)

Facts below come from `pip freeze` / `conda list` / `scripts/check_environment.py` output recorded on the machine described here.

Windows 11, conda environment, Python 3.12.14, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA reported available by torch).

| Package | Version | Installed by |
|---|---|---|
| torch | 2.13.0+cu130 (`cp312-win_amd64`, from `https://download.pytorch.org/whl/cu130`) | pip |
| chemprop | 2.3.1 | pip |
| lightning / pytorch-lightning | 2.6.6 / 2.6.6 | pip |
| torchmetrics | 1.9.0 | pip |
| lightgbm | 4.7.0 | pip |
| rdkit | 2026.3.6 | conda-forge |
| numpy / scipy / pandas | 2.5.3 / 1.18.1 / 3.0.6 | conda-forge (numpy, pandas), pip metadata (scipy) |
| scikit-learn | 1.9.1 | pip metadata |
| PyYAML / pytest | 6.0.3 / 9.1.1 | pip |
| mkl (+ libblas/liblapack `*_mkl`) | 2026.1.0 | conda-forge |
| tabicl | 2.2.0 (`--no-deps`, then einops 0.8.2, huggingface-hub 2.2.0); checkpoint `tabicl-classifier-v2-20260212.ckpt` in the HF cache | pip |
| conda env var | `MKL_THREADING_LAYER=SEQUENTIAL` (single OpenMP runtime with conda-forge MKL + pip torch) | conda |

What this log established: `import torch` succeeded in a fresh process (`check_environment.py`), CUDA was available, and the same torch also
passed `test_real_torch_seeding` when that test was run alone. It failed only inside the full pytest session, after other native libraries
(numpy/scikit-learn MKL-backed numerics, among others) had run in the same process. That pattern argues against a wrong wheel, a CUDA mismatch or
a missing VC++ runtime (those fail in a fresh process too) and for an import-order / DLL-shadowing conflict. The responsible DLL has not been
identified: run `python scripts/check_environment.py --order-probes`.
Not independently verified: that these exact versions are mutually compatible beyond what the log shows.
