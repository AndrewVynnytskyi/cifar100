# CIFAR-100 VGG16 Project

A CIFAR-100 image classification project built around a hand-written VGG16 backbone, intended
to eventually cover a baseline classifier plus three additional experiments (hierarchical
fine/coarse classification, supervised contrastive pretraining, and iterative filter pruning).
Personal learning project, also intended as a portfolio piece.

## Status

**Work in progress, not runnable end-to-end yet.**

- `src/model.py`: `FINE_TO_COARSE` mapping and `VGG16CIFAR100`'s conv blocks (`conv_l1`
  through `conv_l5`, `avgpool`, `classifier`) are written. `VGG16CIFAR100.forward()`,
  `VGG16DualHead`, and `build_projection_matrix()` are not written yet.
- `src/train.py`: `Cutout` and `build_dataloaders` are written. The `Trainer` class
  (`__init__`, `_train_one_epoch`, `_evaluate`, `_save_checkpoint`, `fit`) is not written yet.
- `src/experiments/` (`hierarchical.py`, `supcon.py`, `pruning.py`) — not started, files do not
  exist yet.
- `notebooks/06_cifar100_baseline.ipynb` through `09_pruning_experiment.ipynb`, referenced in
  `PLAN.md`, do not exist yet.
- `tests/` contains only `__init__.py` — no tests written yet.
- Nothing has been run end-to-end. No results exist.

See `PLAN.md` for the full task breakdown and what's planned next.

## Structure

```
src/
  model.py                    FINE_TO_COARSE, VGG16CIFAR100 (conv blocks written;
                               forward(), VGG16DualHead, build_projection_matrix() pending)
  train.py                    Cutout, build_dataloaders (Trainer pending)
  experiments/                empty package, no experiment modules yet
notebooks/
  _reference_custom_cnn.ipynb pre-existing scratch notebook (custom CNN + a self-written
                               VGG16 this project's VGG16CIFAR100 was ported from) — kept
                               for reference, not part of the current workflow
  _reference_resnet.ipynb     pre-existing scratch notebook (ResNet) — kept for reference
tests/
  __init__.py                 empty, no tests yet
configs/
  train.yaml                  hyperparameter config; references Hydra/OmegaConf/W&B, but no
                               code in this repo loads or uses it yet
Dockerfile                    multi-stage build referencing src.train — untested, src.train
                               has no __main__/entry point yet
PLAN.md                       task breakdown for what remains to be built
```

## How to run

No entry point in this repo is runnable yet — `src/train.py` has no `Trainer` and no
`if __name__ == "__main__"` block, so `python -m src.train` (referenced in the Dockerfile
`CMD`) does not work yet.

Untested / not yet possible to verify:
- `pip install -e ".[dev]"` — dependencies are declared in `pyproject.toml` but the install
  itself has not been run in this repo.
- `pre-commit install` / `pre-commit run --all-files` — configured in
  `.pre-commit-config.yaml`, not yet run.
- `docker build .` — the `Dockerfile` exists but has not been built or run.
- Any notebook — none of the planned notebooks exist yet.
