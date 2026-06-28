# Learning Skeleton Version

This folder is a learning-focused skeleton version of the original facial expression recognition project.

The goal is **not** to replace the original project. The goal is to help you study the project architecture first, then gradually understand the core code.

## What this folder keeps

This version keeps the main deep learning application flow:

```text
Dataset
↓
DataLoader
↓
CNN model
↓
Training
↓
Evaluation
↓
Model saving
↓
Model loading
↓
Single image prediction
↓
Streamlit demo
↓
Basic tests
```

## What this folder removes or reduces

This version removes or simplifies details that can distract from the main learning line:

- long print formatting
- excessive comments
- repeated checks
- secondary helper logic
- non-essential output decoration

It keeps the key steps that you must understand:

- model definition
- image transform
- Dataset / DataLoader
- loss function
- optimizer
- forward pass
- backward pass
- parameter update
- evaluation mode
- no_grad
- model save/load
- prediction pipeline

## How to use this folder

Recommended learning order:

```text
1. Read this README
2. Read src/model.py
3. Read src/data_loader.py
4. Read src/train.py
5. Read src/evaluate.py
6. Read src/load_model.py
7. Read src/predict.py
8. Read app/streamlit_app.py
9. Read tests only after you understand the main flow
```

## Learning priority

### A. Must understand now

- What each file is responsible for
- Data flow from image folder to CNN output
- CNN input and output shape
- Training loop order: forward → loss → zero_grad → backward → step
- Evaluation loop: eval + no_grad + argmax + accuracy
- Save/load model weights with state_dict

### B. Know roughly for now

- pathlib path handling
- Streamlit page layout
- pytest structure
- temporary test data

### C. Can skip for now

- advanced model architectures
- logging systems
- configuration files
- best checkpoint saving
- training curves
- confusion matrix
- face detection and alignment

## How to verify your understanding

After reading each file, answer:

1. What is this file responsible for?
2. What are its inputs and outputs?
3. Which file uses it?
4. Which lines are the core logic?
5. If this file disappeared, which later step would break?
