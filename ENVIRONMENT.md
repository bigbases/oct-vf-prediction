# Execution environment

The reported numbers were produced with the versions below. The image branch was
trained on a GPU machine; everything else runs on CPU.

| Component | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch | 1.12.1 + CUDA 11.3 |
| torchvision | 0.13.1 |
| timm | 1.0.27 |
| XGBoost | 3.2.0 |
| NumPy | 1.26.4 |
| SciPy | 1.15.2 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.9.4 |
| Platform | Linux |

Exact pins are in `requirements.txt`; `environment.yml` is the equivalent conda
specification. `extract_cirrus.py` and the `ocr_*.py` scripts additionally need
the `tesseract` and `poppler-utils` system binaries.

## Install

```bash
conda env create -f environment.yml && conda activate hvf
# or
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make check      # config consistency + tests + the PHI scan
```

The conda environment used for the reported runs needs its own libstdc++ ahead of
the system one; if an import fails with `GLIBCXX_3.4.26 not found`, set

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## The `env:` line in module docstrings

Many scripts end their docstring with `env: hvf` or `env: aaa`. These are the two
conda environments the work was done in:

* `hvf` — the environment `environment.yml` describes: torch, timm, xgboost and
  the numerical stack. Everything that trains or evaluates a model runs here.
* `aaa` — the same numerical stack plus matplotlib and openpyxl. The figure
  scripts and the manual-review sheet run here, because the training environment
  was built without matplotlib.

A single environment created from `environment.yml` covers both; the tags record
which one a script was actually run in.

## Paths

Every filesystem root is resolved from the environment by `paths.py`; no absolute
path is baked into the code.

| Variable | Meaning | Default |
|---|---|---|
| `HVF_ROOT` | This repository | the repository directory |
| `HVF_DATA_ROOT` | Parent of `cirrus_out/`, the withheld report archive | `HVF_ROOT` |
| `HVF_LEGACY_ROOTS` | Colon-separated older archive roots, searched in order | unset |
| `HVF_POPPLER_PATH` | `poppler-utils` binaries, if not on `PATH` | unset |
| `HVF_CONFIG` | Overrides `config/params.yaml` | `config/params.yaml` |

## Checking the versions

```bash
python --version
python -c "import torch, timm, xgboost; print(torch.__version__, timm.__version__, xgboost.__version__)"
pip freeze
```

## The pins are deliberate and are not kept current

They record the environment the reported numbers were produced in, so dependency
scanners will flag advisories against them. Upgrading is expected to move the
numbers: the fusion weight is fitted to four decimal places and the reported
differences are of the order of 0.01 dB. Anyone reusing this code on new data
should upgrade and re-validate; anyone reproducing the reported values should
install exactly these versions.
