# Dataset

## 1. Dataset Overview

This project uses the **MIT-BIH Arrhythmia Dataset** for classifying ECG heartbeats into five different classes.

The dataset contains ECG heartbeat signals represented as numerical values. Each heartbeat is associated with a class label that identifies the type of heartbeat.

The dataset is suitable for training a Recurrent Neural Network (RNN) because ECG signals are sequential time-series data.

## 2. Dataset Files

The dataset contains the following files:

- mitbih_train.csv
  - 87,554 samples
  - 188 columns
  - 187 ECG signal values and 1 class label

- mitbih_test.csv
  - 21,892 samples
  - 188 columns
  - 187 ECG signal values and 1 class label

- ptbdb_abnormal.csv
  - 10,506 samples
  - 188 columns
  - Contains abnormal ECG signals

- ptbdb_normal.csv
  - 4,046 samples
  - 188 columns
  - Contains normal ECG signals

The main experiment uses mitbih_train.csv and mitbih_test.csv because they contain the five heartbeat classes required for multi-class classification.

## 3. Data Structure

Each MIT-BIH record contains 188 values.

The first 187 values represent the ECG signal.

The last value represents the heartbeat class label.

The input data is therefore structured as:

187 ECG signal values → 1 heartbeat label

For RNN training, the input is reshaped into:

(samples, 187, 1)

Here, 187 represents the sequence length and 1 represents the ECG signal feature at each time step.

## 4. Heartbeat Classes

The MIT-BIH dataset contains five heartbeat classes:

0 - Normal

1 - Supraventricular ectopic beat

2 - Ventricular ectopic beat

3 - Fusion beat

4 - Unknown beat

These labels are used as the target classes for the classification model.

## 5. Class Distribution

The training dataset contains:

Class 0: 72,471 samples

Class 1: 2,223 samples

Class 2: 5,788 samples

Class 3: 641 samples

Class 4: 6,431 samples

The testing dataset contains:

Class 0: 18,118 samples

Class 1: 556 samples

Class 2: 1,448 samples

Class 3: 162 samples

Class 4: 1,608 samples

The dataset is highly imbalanced because normal heartbeats occur much more frequently than some abnormal heartbeat types.

## 6. ECG Signal Representation

Each heartbeat is represented as a sequence of 187 numerical ECG values.

The sequence represents changes in the electrical activity of the heart over time.

The RNN processes these values sequentially to learn patterns that can help distinguish between different heartbeat classes.

## 7. Data Preprocessing

The following preprocessing steps are applied before training:

1. Separate ECG signal values from the class labels.

2. Normalize the ECG signal values to improve model training.

3. Reshape the ECG signals into the format required by the RNN.

4. Convert the class labels into the required classification format.

5. Use the provided training and testing datasets separately.

6. Handle class imbalance during model training when required.

## 8. Training and Testing Data

The MIT-BIH dataset provides separate training and testing files.

The training data is used to learn patterns from ECG signals.

The testing data is used to evaluate the model on previously unseen ECG signals.

Keeping the training and testing data separate helps provide a more reliable evaluation of model performance.

## 9. Why the Dataset Is Suitable for RNN

ECG signals are time-series data because the values occur in a specific sequence over time.

RNNs are designed to process sequential data and learn relationships between values at different time steps.

For this project, the RNN receives the ECG heartbeat as a sequence of 187 values and predicts its heartbeat class.

## 10. Dataset Challenges

The main challenge is class imbalance.

Normal heartbeat samples are much more common than some abnormal heartbeat classes.

This imbalance can cause the model to perform well on the majority class while performing poorly on minority classes.

Therefore, evaluation should consider metrics such as precision, recall, F1-score, and confusion matrix instead of relying only on accuracy.
