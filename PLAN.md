# CIFAR-100 VGG16 Project — Plan

Phase 3 of `LEARNING_ROADMAP.md` (MNIST -> CIFAR-10 -> CIFAR-100). Division of labor between
Claude (scaffolding / `[COPY-PASTE]` tier only) and you (`[WRITE FROM SCRATCH]` — the actual
learning target).

## Done by Claude

- [x] Repo restructured: `notebooks/`, `src/`, `src/experiments/`, `configs/`, `tests/`
      (pre-roadmap MNIST/CIFAR-10 scratch files removed — that work lives in the sibling
      `mnist-project` and `Cifar10-project` repos; old CIFAR-100 notebooks kept as
      `notebooks/_reference_*.ipynb`)
- [x] `pyproject.toml`, `.pre-commit-config.yaml` — aligned with `mnist-project` (ruff + mypy,
      was black+flake8)
- [x] `Dockerfile` (multi-stage, matches `mnist-project` pattern) — untested, no entry point yet
- [x] `configs/train.yaml` (100 classes, 20 superclasses, SGD+OneCycleLR defaults,
      `label_smoothing=0.1`) — not yet loaded by any code in this repo
- [x] `README.md` reflecting actual current repo state (status, structure, what runs/doesn't)
- [x] `.gitignore`: added `.idea/`, `outputs/`, `checkpoints/`, `wandb/`
- [x] `LICENSE`: MIT, corrected to this project's author
- [x] `src/model.py` — `FINE_TO_COARSE` (100 entries, derived from the actual
      `cifar-100-python/train` pickle, not retyped from memory)
- [x] `src/train.py` — `Cutout` transform + `build_dataloaders` (CIFAR-100 mean/std,
      RandomCrop+Flip, Cutout applied AFTER Normalize)

Nothing else in `src/` or `notebooks/06-09` has been written.

## Yours — strict order, each `[WRITE FROM SCRATCH]`

### 3.1 — `src/model.py`
- [ ] `build_projection_matrix()` (4 lines)
- [x] `VGG16CIFAR100.__init__` — conv blocks ported from `notebooks/_reference_custom_cnn.ipynb`,
      fixed (removed the erroneous extra MaxPool2d in the last stage, fixed 1x1 conv padding,
      added AdaptiveAvgPool2d + a CIFAR-sized classifier head)
- [ ] `VGG16CIFAR100.forward` (assemble the ported blocks — write this yourself)
- [ ] `VGG16DualHead.__init__` / `forward` (shared trunk, fine+coarse heads)

### 3.2 — `src/train.py`
- [ ] `Trainer.__init__` (SGD+Nesterov, label smoothing, OneCycleLR)
- [ ] `Trainer._train_one_epoch` (+ grad clipping, `scheduler.step()` PER BATCH)
- [ ] `Trainer._evaluate` (extend with top-5)
- [ ] `Trainer._save_checkpoint` / `fit`

### 3.3 — `notebooks/06_cifar100_baseline.ipynb`
- [ ] Baseline run WITHOUT `label_smoothing`, then WITH — compare curves

### 3.4 — `src/experiments/hierarchical.py` + `notebooks/07_hierarchical_experiment.ipynb`
- [ ] `CIFAR100WithCoarse` Dataset (`__init__`/`__len__`/`__getitem__`)
- [ ] `HierarchicalLoss` (fine + coarse + consistency KL, `+1e-8` before `.log()`)
- [ ] `HierarchicalTrainer._train_one_epoch` (3-tuple unpacking, tuple model output)

### 3.5 — `src/experiments/supcon.py` + `notebooks/08_supcon_pretrain.ipynb`
- [ ] `TwoViewTransform`, `SupConVGG16`
- [ ] `SupConLoss.forward` — most complex method in all 3 projects, budget a full afternoon

### 3.6 — `src/experiments/pruning.py` + `notebooks/09_pruning_experiment.ipynb`
- [ ] `compute_filter_importance`
- [ ] `prune_model_filters` (global quantile threshold)

## Handoff pattern

Write the WRITE-FROM-SCRATCH core for a section, then ping me — I drop in the paired
`[COPY-PASTE]` boilerplate for that section (dataloaders, orchestration wrappers, trivial
utils) so you're never retyping mechanical code.
