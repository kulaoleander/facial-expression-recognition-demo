# Facial Expression Recognition Demo

An end-to-end deep learning and computer vision project for facial expression recognition.

This project predicts one of seven facial expressions from face images and includes a complete AI application pipeline: dataset inspection, PyTorch data loading, CNN baselines, transfer learning with ResNet18, configurable training, class imbalance handling, learning rate scheduling, early stopping, final test evaluation, model saving/loading, single-image prediction, OpenCV face detection, Streamlit demo, and pytest-based automated tests.

The goal of this project is not to claim state-of-the-art performance, but to build a complete, explainable, reproducible AI application that can be shown on GitHub, discussed in interviews, and extended into real-time computer vision applications.

---

## Project Overview

The project classifies facial expressions into seven categories:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

Input format:

```text
grayscale image
48x48 pixels
tensor shape: [batch_size, 1, 48, 48]
```

Model output format:

```text
logits shape: [batch_size, 7]
```

The project started with a simple CNN baseline and was gradually improved with validation/test split, data augmentation, ImprovedCNN, transfer learning with ResNet18, robust prediction outputs, OpenCV-based face crop support, final test set evaluation, and a more reproducible training pipeline.

Version 1.2 upgrades the training workflow with command-line training configuration, automatic CPU/CUDA device selection, class weights for imbalanced classes, learning rate scheduling, early stopping, GPU training, and expanded pytest coverage.

---

## Main Features

- FER2013-style dataset structure inspection
- PyTorch `ImageFolder` data loading
- Train / validation / test split
- Data augmentation for training data
- SimpleCNN baseline model
- ImprovedCNN with BatchNorm and Dropout
- ResNet18 transfer learning model
- Configurable command-line training script
- Automatic CPU/CUDA device selection
- Class weights for imbalanced training data
- Learning rate scheduler support
- Early stopping support
- Training loop with `CrossEntropyLoss` and Adam optimizer
- Validation accuracy tracking
- Best model checkpoint saving
- Final test set evaluation
- Training configuration saved as JSON
- Training history saved as JSON
- Training curves saved as PNG
- Experiment comparison logging as CSV
- Detailed evaluation with classification report and confusion matrix
- Single-image prediction
- Top-3 prediction probabilities
- Low-confidence warning
- OpenCV Haar Cascade face detection
- Largest face crop with fallback to original image
- Streamlit web demo
- Automated tests with pytest

---

## Tech Stack

- Python
- PyTorch
- torchvision
- OpenCV
- Pillow
- NumPy
- pandas
- matplotlib
- scikit-learn
- Streamlit
- pytest

---

## Dataset

This project is designed for a FER2013-style facial expression dataset organized as image folders.

Expected local dataset structure:

```text
data/raw/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The dataset is not committed to GitHub because image datasets are usually large.

---

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the environment check:

```powershell
python -m src.environment_check
```

---

## Dataset Inspection

Run:

```powershell
python -m src.inspect_dataset
```

This checks:

- train/test folder structure
- emotion class folders
- image counts
- sample image mode
- sample image size

Expected sample image format:

```text
mode: L
size: (48, 48)
```

---

## Train the Model

Basic training command:

```powershell
python -m src.train
```

The training script also supports configurable command-line arguments.

Example formal ResNet18 training command:

```powershell
python -m src.train --model resnet18 --epochs 50 --batch-size 32 --lr 0.001 --pretrained --use-augmentation --class-weights --scheduler reduce_on_plateau --scheduler-patience 4 --scheduler-factor 0.5 --early-stopping-patience 8 --device cuda
```

Main training options:

```text
--model simple_cnn | improved_cnn | resnet18
--epochs
--batch-size
--lr
--pretrained / --no-pretrained
--use-augmentation / --no-use-augmentation
--class-weights / --no-class-weights
--scheduler none | reduce_on_plateau | cosine
--scheduler-patience
--scheduler-factor
--early-stopping-patience
--device auto | cpu | cuda
```

The formal v1.2 training run used:

```text
Model: resnet18
Pretrained weights: True
Epochs: 50
Batch size: 32
Learning rate: 0.001
Data augmentation: True
Class weights: True
Scheduler: reduce_on_plateau
Scheduler patience: 4
Scheduler factor: 0.5
Early stopping patience: 8
Device: CUDA / Tesla T4
```

Generated outputs include:

```text
outputs/models/resnet18.pth
outputs/logs/training_config.json
outputs/logs/training_history.json
outputs/logs/experiment_results.csv
outputs/figures/training_curves.png
```

---

## Model Comparison

A controlled experiment was run using the same validation split and training configuration where applicable.

| Model / Version | Validation Accuracy |
|---|---:|
| SimpleCNN baseline | 0.4142 |
| ImprovedCNN | 0.4644 |
| ResNet18 from scratch | 0.4715 |
| Pretrained ResNet18, v1.1 short training | 0.5078 |
| Pretrained ResNet18, v1.2 GPU training + class weights + scheduler | 0.6314 |

The v1.2 pretrained ResNet18 model achieved the best validation accuracy in this project.

This shows the value of transfer learning and training engineering: instead of learning visual features from scratch, the model reuses general image features learned from large-scale image data and fine-tunes them for facial expression recognition. The v1.2 training improvements also made the experiment more reproducible and improved minority-class handling.

---

## Final Test Evaluation

After selecting the best model based on validation accuracy, the v1.2 pretrained ResNet18 model was evaluated once on the held-out test set.

| Metric | Value |
|---|---:|
| Test samples | 7178 |
| Test accuracy | 0.6257 |
| Macro average F1-score | 0.5996 |
| Weighted average F1-score | 0.6224 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| angry | 0.5532 |
| disgust | 0.7027 |
| fear | 0.3701 |
| happy | 0.8005 |
| neutral | 0.6667 |
| sad | 0.4747 |
| surprise | 0.8063 |

Compared with the earlier v1.1 result, the v1.2 training run improved test accuracy from 0.5059 to 0.6257 and macro average F1-score from 0.3993 to 0.5996. The `disgust` class improved substantially after adding class weights, showing that class imbalance handling made the model pay more attention to minority classes.

The final test result still shows that `fear` and `sad` remain relatively challenging classes, which is realistic for low-resolution FER2013-style facial expression recognition.

The final test evaluation outputs are saved locally to:

```text
outputs/logs/final_test_metrics.json
outputs/figures/final_test_confusion_matrix.png
```

---

## Final Test Evaluation Script

Run:

```powershell
python -m src.final_test_evaluation
```

This script:

- loads the default best model
- creates the test DataLoader
- evaluates the model on the held-out test set
- calculates test accuracy, macro F1-score, weighted F1-score, per-class accuracy, and confusion matrix
- saves the final test metrics to JSON
- saves the final test confusion matrix as PNG

The default best model is:

```text
outputs/models/resnet18.pth
```

---

## Single Image Prediction

Run:

```powershell
python -m src.predict
```

By default, prediction loads the ResNet18 model from `outputs/models/resnet18.pth`.

The prediction output includes:

- predicted class
- confidence
- top-3 predictions
- low-confidence flag
- face crop status
- number of detected faces
- selected face box

If no face is detected, the system falls back to the original image instead of crashing.

---

## Streamlit Demo

Run:

```powershell
streamlit run app/streamlit_app.py
```

By default, the Streamlit demo uses the ResNet18 model from `outputs/models/resnet18.pth`.

The web app allows users to:

- upload an image
- preview the uploaded image
- enable or disable face detection / face crop
- see whether a face was detected
- view the image used for prediction
- run emotion prediction
- see predicted emotion and confidence
- see top-3 prediction probabilities
- see a low-confidence warning when appropriate

---

## Face Detection and Face Crop

The project uses OpenCV Haar Cascade face detection.

The face crop pipeline is:

```text
Uploaded image
↓
OpenCV face detection
↓
Select largest detected face
↓
Add padding around face box
↓
Crop face region
↓
Resize to 48x48
↓
Run expression prediction
```

If no face is detected:

```text
No detected face
↓
Use original image as fallback
↓
Continue prediction
```

This makes the demo more robust for real-world uploaded images.

---

## Run Tests

Run all tests:

```powershell
python -m pytest
```

Current local test result:

```text
106 passed, 1 skipped
```

Current test coverage includes:

- environment check
- dataset inspection
- DataLoader output shape and split logic
- data augmentation behavior
- model output shape
- SimpleCNN / ImprovedCNN / ResNet18 model creation
- configurable training argument parsing
- automatic CPU/CUDA device selection
- class weight calculation
- weighted loss function creation
- learning rate scheduler creation
- early stopping logic
- training loop
- model saving
- model loading
- ResNet18 loading consistency
- training configuration saving
- training history saving
- training curve saving
- experiment summary saving
- evaluation accuracy
- detailed evaluation metrics
- final test evaluation metrics
- single-image prediction
- top-k prediction logic
- low-confidence logic
- OpenCV face detection utilities
- face crop fallback behavior

---

## Project Pipeline

```text
Dataset
↓
Dataset inspection
↓
DataLoader
↓
Train / validation / test split
↓
Data augmentation
↓
CNN baseline
↓
Improved CNN
↓
Transfer learning with ResNet18
↓
Configurable training
↓
Class weights for imbalance handling
↓
Learning rate scheduling
↓
Early stopping support
↓
Best model checkpoint saving
↓
Final test evaluation
↓
Single-image prediction
↓
Top-3 prediction and low-confidence warning
↓
OpenCV face detection and face crop
↓
Streamlit web demo
↓
Automated tests
```

---

## What I Learned

This project helped me practice:

- building a complete PyTorch image classification pipeline
- using train / validation / test split correctly
- comparing models with controlled experiments
- using transfer learning with ResNet18
- handling class imbalance with class weights
- using learning rate scheduling during training
- using early stopping to make longer training safer
- evaluating classification models beyond overall accuracy
- using a held-out test set for final model evaluation
- building a simple AI web demo with Streamlit
- adding OpenCV preprocessing before model inference
- writing pytest tests for machine learning project components
- keeping training, evaluation, prediction, and demo model loading consistent
- organizing a project for GitHub and interview discussion

---

## Limitations

Current limitations:

- FER2013-style 48x48 grayscale images are low-resolution.
- Facial expression labels can be ambiguous.
- Some classes are harder due to imbalance and visual similarity.
- `fear` and `sad` remain relatively challenging in the final test evaluation.
- Haar Cascade face detection is simple and may fail on low-resolution or non-frontal faces.
- Softmax confidence is not a fully calibrated probability.
- The project focuses on building a complete applied AI pipeline rather than achieving state-of-the-art performance.

---

## Possible Future Improvements

Possible future improvements include:

- use a stronger face detector
- add face alignment
- tune learning rate and batch size further
- try stronger backbones
- improve class imbalance handling further
- add model selection in the Streamlit interface
- deploy the Streamlit demo online
- extend the project to webcam-based real-time prediction
- combine expression recognition with attention-state estimation

---

## Project Status

Version 1.2 completed.

The project now supports a complete facial expression recognition workflow from dataset loading to model training, validation-based model selection, final test evaluation, robust prediction, face crop preprocessing, web demo, and automated tests.

Version 1.2 adds a reproducible training pipeline with command-line configuration, automatic CPU/CUDA device selection, class weights, learning rate scheduling, early stopping, GPU training, improved final test performance, and expanded automated tests.