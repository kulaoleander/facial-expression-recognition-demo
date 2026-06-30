# Experiment Comparison: SimpleCNN vs ImprovedCNN

## 1. Purpose

The purpose of this experiment is to compare the baseline CNN model with an improved CNN model for facial expression recognition.

This project uses a FER2013-style dataset with 7 facial expression classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The comparison focuses on validation accuracy, because the test set should be reserved for final evaluation only.

---

## 2. Experiment Setup

Both models were trained under the same experimental settings.

| Setting | Value |
|---|---|
| Dataset | FER2013-style facial expression dataset |
| Input shape | 1 × 48 × 48 grayscale image |
| Number of classes | 7 |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Validation ratio | 0.2 |
| Data augmentation | Enabled for training set only |
| Validation augmentation | Disabled |

The training set was used for parameter updates.

The validation set was used to monitor model performance during training.

The test set was not used during training or model selection.

---

## 3. Models Compared

### 3.1 SimpleCNN

SimpleCNN is the baseline model.

It contains:

- 2 convolution blocks
- ReLU activations
- Max pooling
- A simple fully connected classifier

Its role is to provide a simple reference point for comparison.

### 3.2 ImprovedCNN

ImprovedCNN is a stronger CNN model.

Compared with SimpleCNN, it includes:

- More convolution channels
- An additional convolution block
- BatchNorm layers
- Dropout in the classifier

Its role is to test whether a deeper and more regularized CNN can improve validation performance.

---

## 4. Results

| Model | Final Validation Accuracy | Best Validation Accuracy |
|---|---:|---:|
| SimpleCNN | 0.4142 | 0.4142 |
| ImprovedCNN | 0.4644 | 0.4644 |

ImprovedCNN achieved higher validation accuracy than SimpleCNN.

The absolute improvement was:

0.4644 - 0.4142 = 0.0502

This means ImprovedCNN improved validation accuracy by about 5.02 percentage points compared with the baseline model.

---

## 5. Interpretation

The result suggests that the improved model structure is beneficial for this task.

The likely reasons are:

1. ImprovedCNN has more convolution channels, allowing it to learn richer image features.
2. ImprovedCNN has an additional convolution block, allowing it to learn more complex patterns.
3. BatchNorm may help make training more stable.
4. Dropout may help reduce overfitting risk.

However, the experiment only used 3 epochs, so this result should be treated as an early comparison rather than a final conclusion.

---

## 6. Current Conclusion

ImprovedCNN performed better than SimpleCNN under the current training setup.

The current best model is:

ImprovedCNN

The current best validation accuracy is:

0.4644

Therefore, ImprovedCNN should be used as the stronger model for the next stage of the project.

---

## 7. Next Steps

Possible next improvements include:

1. Train for more epochs.
2. Save the best model based on validation accuracy.
3. Add final test set evaluation for the selected model.
4. Try transfer learning with ResNet18.
5. Compare model performance using per-class accuracy and confusion matrix.