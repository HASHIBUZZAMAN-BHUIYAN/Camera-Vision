# Setup Guide

## After a fresh clone

The `.venv` folder is intentionally **not committed** to this repo (it's listed in `.gitignore`). Every fresh clone needs the environment rebuilt once — this is normal and by design, since virtual environments are machine-specific.

### One-command setup

Run this from the repo root:

```
.\setup_venv.bat
```

This will:
1. Create `.venv` (if it doesn't already exist)
2. Activate it
3. Install all pinned dependencies from `requirements.txt`

Requires **Python 3.11+** on your PATH.

---

## Running the projects

Both projects share the same `.venv` at the repo root.

**Neural Network Visualization** — hand-gesture digit recognition with live neural network view:
```
cd "Neural Network Visualization"
.\..\. venv\Scripts\activate
python app.py
```

**Vision Puzzle** — hand-gesture photo-slicing jigsaw puzzle:
```
cd "Vision puzzle"
.\..\. venv\Scripts\activate
python app.py
```

Controls are shown on-screen when each app launches.

---

## Dependencies summary

| Package | Version | Used by |
|---|---|---|
| opencv-python | 4.13.0.92 | Both |
| mediapipe | 0.10.9 | Both |
| numpy | 2.4.6 | Both |
| scikit-learn | 1.9.0 | Neural Network Visualization |
| joblib | 1.5.3 | Neural Network Visualization |

Full pinned manifest: `requirements.txt`

---

## Re-training the model (optional)

The pre-trained `model.pkl` and `scaler.pkl` are already included in `Neural Network Visualization/`. If you want to retrain:

```
cd "Neural Network Visualization"
python train_model.py
```

This downloads the MNIST dataset (~11 MB) and takes ~1 minute to train.
