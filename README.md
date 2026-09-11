**ECG Heartbeat Classification using RNN**

**Overview**

This project focuses on classifying ECG heartbeat signals into different heartbeat categories using a Recurrent Neural Network (RNN).

Electrocardiogram (ECG) signals contain important information about the electrical activity of the heart. Different heartbeat patterns can indicate normal or abnormal cardiac activity. The aim of this project is to use deep learning to automatically learn patterns from ECG signals and classify them into their corresponding categories.

The project uses a **SimpleRNN** model to learn temporal patterns present in ECG heartbeat signals.

**Project Objective**

The main objective of this project is to build a deep learning model that can:

* Process ECG heartbeat signals
* Learn important patterns from the signals
* Classify individual heartbeats into different classes
* Evaluate the performance of the trained model
* Make predictions on new ECG heartbeat samples

**Project Workflow**

The project follows this workflow:

```text
ECG Dataset
     ↓
Data Cleaning
     ↓
Data Preprocessing
     ↓
Train-Test Split
     ↓
Data Normalization
     ↓
RNN Model
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Prediction
```

## Model Used

The main deep learning model used in this project is:

**Simple Recurrent Neural Network (SimpleRNN)**

The model contains:

* Input layer
* SimpleRNN layer
* Fully connected Dense layer
* Output layer with Softmax activation

The RNN is used because ECG data is sequential in nature and the order of signal values contains useful information for classification.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn

## Documentation

Detailed project information is maintained separately:

* **Problem Statement:** `docs/problem_statement.md`
* **Dataset Information:** `docs/dataset.md`

## Project Structure

```text
ECG-Heartbeat-Classification-RNN/
│
├── README.md
│
├── docs/
│   ├── problem_statement.md
│   └── dataset.md
│
├── data/
│   └── README.md
│
├── train.py
├── predict.py
├── requirements.txt
└── .gitignore
```

Additional folders such as `models/`, `results/`, and `app/` may be added as the project develops.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ECG-Heartbeat-Classification-RNN.git
```

Move into the project directory:

```bash
cd ECG-Heartbeat-Classification-RNN
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

Run:

```bash
python train.py
```

This will load the dataset, preprocess the ECG signals, train the RNN model, and evaluate its performance.

### Make Predictions

After training the model, predictions can be made using:

```bash
python predict.py
```

## Model Evaluation

The model will be evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Training and validation performance will also be visualized using accuracy and loss curves.

Actual results will be added after completing model training and evaluation.

## Future Improvements

Possible improvements include:

* Experimenting with LSTM and GRU networks
* Handling class imbalance more effectively
* Hyperparameter tuning
* Comparing different deep learning architectures
* Improving model performance
* Adding a web-based prediction interface
* Deploying the trained model as an application

## Limitations

This project is intended for educational and research purposes. The model should not be considered a medical diagnostic system and should not be used for making real-world medical decisions.

## Learning Objectives

Through this project, the following concepts are explored:

* ECG signal classification
* Data preprocessing
* Feature normalization
* Train-test splitting
* Recurrent Neural Networks
* Model training
* Model evaluation
* Classification metrics
* Deep learning workflow

## Author

**Divya Kumari**

This project is developed as part of my learning and portfolio development in **Artificial Intelligence, Machine Learning, and Deep Learning**.

## License

This project is intended for educational and research purposes.
