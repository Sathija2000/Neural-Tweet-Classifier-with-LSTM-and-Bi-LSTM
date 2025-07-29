# Tweet Classification using LSTM and Bi-LSTM

This project evaluates and compares the performance of Long Short-Term Memory (LSTM) and Bidirectional LSTM (Bi-LSTM) models for tweet classification. The task involves detecting whether a tweet is a **personal health mention** (label 1) or **non-mention** (label 0).

##  Overview

Tweets are short, informal, and context-dependent, making them suitable for sequence models like LSTM and Bi-LSTM, which can capture temporal and contextual patterns. This project implements and compares both models on preprocessed tweet data.

##  Models

- **LSTM Model**  
  Two stacked LSTM layers with dropout for better generalization.

- **Bi-LSTM Model**  
  A bidirectional LSTM layer followed by a second LSTM layer.

Both models use:
- Trainable embedding layer (dimension = 100)
- `ReLU` activation in output (used with thresholding)
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Epochs: 5
- Batch size: 128

##  Results

| Metric              | LSTM     | Bi-LSTM  |
|---------------------|----------|----------|
| Correct Predictions | 2726     | 2734     |
| Wrong Predictions   | 605      | 597      |
| Accuracy            | 81.84%   | 82.08%   |

##  Key Points

- **Bi-LSTM slightly outperforms** LSTM by leveraging both past and future context.
- **Dropout** and **stacked layers** help in generalization.
- `ReLU` in the output layer works due to custom thresholding, though `sigmoid` is more conventional for binary classification.
- Input length deprecation in TensorFlow was resolved by using `.build(input_shape=(None, 10))` instead of `input_length`.

