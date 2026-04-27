# SortSmart Share Bundle

This folder contains the files needed to share the current state of the CPE 595 Applied AI project with teammates.

## What Is Included

- `app.py`
  Flask demo app for running the trained classifier.
- `sortsmart_pipeline.py`
  Training and evaluation pipeline for the current midterm model.
- `train_deep_models.py`
  PyTorch/timm training runner for modern pretrained vision models.
- `deep_vision.py`
  Shared model, transform, and checkpoint inference utilities.
- `deep_inference.py`
  CLI helper for testing a saved deep-learning checkpoint on one image.
- `output/`
  Saved model artifacts, evaluation plots, and the results summary.
- `templates/`
  HTML template used by the Flask app.
- `static/samples/`
  Sample images used by the demo UI.
- `trashnet-master/data/dataset-resized/`
  TrashNet resized dataset used by the pipeline.
- `CPE 595 Midterm-Team6.pdf`
  Midterm report slides.
- `requirements.txt`
  Python packages needed for the classical app and pipeline.
- `requirements-deep.txt`
  Extra packages for the PyTorch/timm deep-learning track.

## What Is Not Included

- The original repo `.venv`
- The legacy Lua/Torch training code from the upstream TrashNet repo
- Git metadata

This share bundle is focused on the current Python-based project path.

## Project Summary

SortSmart classifies TrashNet waste images into six classes:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

The project now has three model families:

- Classical ML baseline with HOG + HSV features.
- A custom CNN trained from scratch.
- Pretrained deep vision models using PyTorch/timm.

The final website uses only the validation-selected best deep checkpoint:

- Model: `vit_base_patch16_224`
- Checkpoint: `output/deep/best_model.pt`
- Validation Macro-F1: `0.9049`
- Test Accuracy: `0.9263`
- Test Macro-F1: `0.9219`
- Median inference latency: `5.23 ms`

The strongest test-only result was `swin_tiny_patch4_window7_224` with Test Macro-F1 `0.9475`, but it was not selected as the published model because final model selection should be based on validation performance, not test-set performance.

Saved midterm classical baseline results in `output/results_summary.txt`:

- Test Accuracy: `0.7895`
- Test Macro-F1: `0.7833`
- Median inference latency: `7.48 ms`

## Setup

For the final deep-learning website and training tools, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-deep.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-deep.txt
```

## Run The Demo App

From inside this `share` folder:

```bash
SORTSMART_DEEP_CHECKPOINT=output/deep/best_model.pt \
SORTSMART_DEEP_DEVICE=cuda \
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Use CPU if CUDA is unavailable:

```bash
SORTSMART_DEEP_CHECKPOINT=output/deep/best_model.pt \
SORTSMART_DEEP_DEVICE=cpu \
python app.py
```

The app supports both sample images from `static/samples/` and user image uploads. It displays the predicted class, the top predictions, and the confidence score for every class.

## Retrain The Model

From inside this `share` folder:

```bash
python sortsmart_pipeline.py
```

This will:

- load `trashnet-master/data/dataset-resized/`
- split the dataset into train/validation/test
- augment the training split only
- train multiple classical ML models
- save updated artifacts into `output/`

## Deep Vision Training Track

The deep-learning track keeps the midterm HOG + HSV pipeline intact and adds transfer learning with:

- `efficientnetv2_s`
- `convnext_tiny`
- `swin_tiny_patch4_window7_224`
- `vit_base_patch16_224`
- `custom_cnn_scratch`

The runner creates the same deterministic 70/15/15 split, trains multiple ablations, evaluates on validation and test sets, and writes artifacts under `output/deep/`:

- `leaderboard.csv`
- `deep_results_summary.txt`
- `best_model.pt`
- `split_seed42.json`
- `runs/<model>/<experiment>/checkpoint_best.pt`
- `runs/<model>/<experiment>/history.csv`
- `runs/<model>/<experiment>/metrics.json`
- `runs/<model>/<experiment>/confusion_matrix.png`
- `runs/<model>/<experiment>/per_class_f1.png`

Current deep-learning results are in:

- `output/deep/pretrained_core_3ep/`
- `output/deep/pretrained_core_lr1e4_5ep/`
- `output/deep/latest_results_summary.txt`
- `output/deep/latest_leaderboard.csv`

The current published deep checkpoint is `output/deep/best_model.pt`, copied from:

```text
output/deep/pretrained_core_lr1e4_5ep/runs/vit_base_patch16_224/full_finetune/checkpoint_best.pt
```

The combined presentation artifacts that include classical ML, custom CNN, and pretrained deep models are:

- `output/deep/presentable/combined_all_model_results.csv`
- `output/deep/presentable/combined_all_model_results.md`
- `output/deep/presentable/combined_validation_macro_f1_all_models.png`
- `output/deep/presentable/combined_test_macro_f1_available_models.png`
- `output/deep/presentable/combined_best_by_family_test_macro_f1.png`
- `output/deep/presentable/combined_neural_training_curves.png`

Recommended workflow:

1. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-deep.txt
```

For this machine, `nvidia-smi` reports an RTX 4080 Laptop GPU with a CUDA 12.5 driver. The CUDA 12.4 PyTorch wheel is the safest target for that driver. If using a different machine, use the CUDA wheel recommended by the PyTorch install selector.

If C: drive space is tight in WSL, create the venv outside the mounted project folder:

```bash
python3 -m venv /tmp/sortsmart-venv
source /tmp/sortsmart-venv/bin/activate
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -r requirements-deep.txt
```

2. Run a dependency and data smoke test.

```bash
python3 train_deep_models.py \
  --ablation-plan smoke \
  --models convnext_tiny \
  --limit-per-class 12 \
  --batch-size 4 \
  --num-workers 0 \
  --latency-repeats 5 \
  --no-pretrained
```

3. Run the first real comparison.

```bash
python3 train_deep_models.py \
  --ablation-plan core \
  --epochs 12 \
  --head-epochs 3 \
  --batch-size 16
```

4. Run the full ablation set once the core run is stable.

```bash
python3 train_deep_models.py \
  --ablation-plan full \
  --epochs 16 \
  --head-epochs 3 \
  --learning-rates 3e-4 1e-4 \
  --batch-size 16
```

The full plan trains head-only, full fine-tuning, stronger augmentation, weighted sampling, and no-class-weight variants for each model. Pretrained weights are downloaded the first time each backbone is used.

5. Test the selected checkpoint on an image.

```bash
python3 deep_inference.py \
  --checkpoint output/deep/best_model.pt \
  --image static/samples/glass.jpg
```

## Notes For Teammates

- The website intentionally uses only the best deep checkpoint, not the classical baseline.
- The saved classical model remains in `output/` for comparison and reporting.
- Uploaded images are written to `static/uploads/` during local demo sessions.
- Test metrics are comparable across the neural models because they use the same deterministic dataset split and evaluation code. For the older classical models, only Gradient Boosting has saved test metrics; the other classical rows report validation scores from the midterm pipeline.
