# SortSmart Recycling Classifier

SortSmart is a CPE 595 Applied AI project that classifies waste images into six TrashNet classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

This repo contains the final web app, deep-learning training code, inference utilities, and presentation-ready result artifacts.

## Final Model

The web app uses one final checkpoint:

| Model | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Median Latency |
|---|---:|---:|---:|---:|
| `vit_base_patch16_224` | `0.9049` | `0.9219` | `0.9263` | `5.23 ms` |

The checkpoint should be placed at:

```text
output/deep/best_model.pt
```

## Key Files

- `app.py`: Flask app using the final deep checkpoint and OpenAI assistant endpoint.
- `templates/index.html`: Upload-capable web UI with confidence scores and chatbot.
- `train_deep_models.py`: Deep model training and ablation runner.
- `deep_vision.py`: Model loading and inference utilities.
- `deep_inference.py`: Command-line inference helper.
- `make_presentable_results.py`: Generates combined result tables and figures.
- `output/deep/presentable/`: Final tables, curves, plots, and metrics.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-deep.txt
```

For CPU-only machines, install the CPU PyTorch wheel from the official PyTorch selector instead of the CUDA wheel.

## Run The Web App

With CUDA:

```bash
export OPENAI_API_KEY=your_openai_api_key
export SORTSMART_OPENAI_MODEL=gpt-5-mini
SORTSMART_DEEP_CHECKPOINT=output/deep/best_model.pt \
SORTSMART_DEEP_DEVICE=cuda \
python app.py
```

With CPU:

```bash
export OPENAI_API_KEY=your_openai_api_key
export SORTSMART_OPENAI_MODEL=gpt-5-mini
SORTSMART_DEEP_CHECKPOINT=output/deep/best_model.pt \
SORTSMART_DEEP_DEVICE=cpu \
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The app supports sample images and user uploads. Uploaded images are classified in memory and are not saved to disk. The recycling assistant uses the latest prediction context and requires `OPENAI_API_KEY`; without it, the classifier still works and the assistant returns a setup message.

For Render, set:

```text
SORTSMART_DEEP_CHECKPOINT=/var/data/best_model.pt
SORTSMART_DEEP_DEVICE=cpu
SORTSMART_OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=<your key>
WEB_CONCURRENCY=1
```

## Training

Run the pretrained model comparison:

```bash
python train_deep_models.py \
  --ablation-plan core \
  --models efficientnetv2_s convnext_tiny swin_tiny_patch4_window7_224 vit_base_patch16_224 \
  --epochs 5 \
  --head-epochs 1 \
  --learning-rates 1e-4 \
  --batch-size 16
```

Run the custom CNN from-scratch ablation:

```bash
python train_deep_models.py \
  --ablation-plan scratch_light \
  --models custom_cnn_scratch \
  --epochs 70 \
  --learning-rates 1e-3 \
  --batch-size 32 \
  --label-smoothing 0.05 \
  --no-pretrained
```

After training runs are available locally, regenerate presentation artifacts:

```bash
python make_presentable_results.py
```

Run inference on one image:

```bash
python deep_inference.py \
  --checkpoint output/deep/best_model.pt \
  --image static/samples/glass.jpg
```

## Results

Use these files for the report or slides:

```text
output/deep/latest_results_summary.txt
output/deep/latest_leaderboard.csv
output/deep/presentable/combined_all_model_results.md
output/deep/presentable/combined_all_model_results.csv
output/deep/presentable/combined_validation_macro_f1_all_models.png
output/deep/presentable/combined_test_macro_f1_available_models.png
output/deep/presentable/combined_neural_training_curves.png
output/deep/presentable/best_confusion_matrix.png
output/deep/presentable/best_per_class_f1.png
output/deep/presentable/best_metrics.json
```
