import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.callbacks import Callback

np.random.seed(42)


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor="val_loss", min_delta=0, patience=0,
                 mode="auto", baseline=None, restore_best_weights=False,
                 start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in ("auto", "min", "max"):
            raise ValueError("mode must be 'auto', 'min', or 'max'")

        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        maximize = mode == "max" or (
            mode == "auto" and any(name in monitor.lower()
                                    for name in ("acc", "accuracy", "auc"))
        )
        self.best = (-np.inf if maximize else np.inf) if baseline is None else baseline
        self._is_improvement = (
            (lambda value, best: value > best + self.min_delta)
            if maximize else
            (lambda value, best: value < best - self.min_delta)
        )

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch < self.start_from_epoch:
            return

        current = logs.get(self.monitor)
        if current is None or not np.isfinite(current):
            return

        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.stopped_epoch:
            print(f"\nEarly stopping at epoch {self.stopped_epoch}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

try:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_test.csv"))
    print("MIT-BIH dataset loaded successfully.")

except FileNotFoundError:
    print("Dataset not found, using dummy data for testing...")

    train_df = pd.DataFrame(np.random.rand(1000, 187))
    test_df = pd.DataFrame(np.random.rand(200, 187))

    train_df[187] = np.random.randint(0, 5, 1000)
    test_df[187] = np.random.randint(0, 5, 200)

X = train_df.iloc[:, :-1].values.astype("float32")
y = train_df.iloc[:, -1].values.astype("int32")

X_test = test_df.iloc[:, :-1].values.astype("float32")
y_test = test_df.iloc[:, -1].values.astype("int32")

print("\nTraining data shape:", X.shape)
print("Test data shape:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

max_value = X_train.max()

X_train = X_train / max_value
X_val = X_val / max_value
X_test = X_test / max_value

X_train = X_train.reshape(-1, 187, 1)
X_val = X_val.reshape(-1, 187, 1)
X_test = X_test.reshape(-1, 187, 1)

print("\nAfter preprocessing:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

model = Sequential([
    Input(shape=(187, 1)),
    SimpleRNN(64, activation="tanh"),
    Dense(32, activation="relu"),
    Dense(5, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nRNN TEST RESULTS")
print("Test Loss:", loss)
print("RNN Test Accuracy:", accuracy)

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_classes,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RNN Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig(
    os.path.join(ASSETS_DIR, "training_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(
    os.path.join(ASSETS_DIR, "training_loss.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Supraventricular",
        "Ventricular",
        "Fusion",
        "Unknown"
    ]
)

disp.plot(ax=ax, values_format="d")
plt.title("ECG Heartbeat Classification Confusion Matrix")
plt.savefig(
    os.path.join(ASSETS_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

model.save(
    os.path.join(MODEL_DIR, "ecg_heartbeat_rnn.keras")
)

print("\nModel saved successfully.")
print("Assets saved in:", ASSETS_DIR)
