# SortSmart Share Bundle

This folder contains the files needed to share the current state of the CPE 595 Applied AI project with teammates.

## What Is Included

- `app.py`
  Flask demo app for running the trained classifier.
- `sortsmart_pipeline.py`
  Training and evaluation pipeline for the current midterm model.
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
  Python packages needed for the app and pipeline.

## What Is Not Included

- The original repo `.venv`
- The legacy Lua/Torch training code from the upstream TrashNet repo
- Git metadata

This share bundle is focused on the current Python-based project path.

## Project Summary

The current project uses the TrashNet dataset to classify six classes of waste:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

The active model pipeline uses:

- HOG features
- HSV color histograms
- classical ML model comparison
- `HistGradientBoostingClassifier` as the saved best model

Saved midterm results in `output/results_summary.txt`:

- Test Accuracy: `0.7895`
- Test Macro-F1: `0.7833`
- Median inference latency: `7.48 ms`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Demo App

From inside this `share` folder:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The current demo uses the six sample images in `static/samples/`.

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

## Notes For Teammates

- The current web app is a sample-image demo, not a user-upload app yet.
- The saved model and scaler are already included, so the app can run without retraining.
- If the team decides to implement image uploads next, the main files to update are `app.py` and `templates/index.html`.
