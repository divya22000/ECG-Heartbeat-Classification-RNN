import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# REPRODUCIBILITY

np.random.seed(42)

# 1. LOAD DATA

try:
    train_df = pd.read_csv("mitbih_train.csv")
    test_df = pd.read_csv("mitbih_test.csv")

    print("MIT-BIH dataset loaded successfully.")

except FileNotFoundError:
    print("Dataset not found, using dummy data for testing...")

    # Dummy data
    train_df = pd.DataFrame(np.random.rand(1000, 187))
    test_df = pd.DataFrame(np.random.rand(200, 187))

    # Add 5 class labels
    train_df[187] = np.random.randint(0, 5, 1000)
    test_df[187] = np.random.randint(0, 5, 200)

    print("WARNING: Accuracy obtained from dummy data is NOT meaningful.")


# 2. PREPROCESSING

# Training data
X = train_df.iloc[:, :-1].values.astype("float32")
y = train_df.iloc[:, -1].values.astype("int32")

# Test data
X_test = test_df.iloc[:, :-1].values.astype("float32")
y_test = test_df.iloc[:, -1].values.astype("int32")


# 3. CHECK DATA

print("\nTraining data shape:", X.shape)
print("Test data shape:", X_test.shape)

print("Number of classes:", len(np.unique(y)))
print("Classes:", np.unique(y))



# 4. HANDLE INVALID VALUES


# Replace NaN and infinite values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)


# 5. TRAIN / VALIDATION SPLIT

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. NORMALIZATION

# Calculate normalization values ONLY
# from the training data.

train_min = np.min(X_train)
train_max = np.max(X_train)

# Prevent division by zero
if train_max - train_min != 0:

    X_train = (X_train - train_min) / (train_max - train_min)
    X_val = (X_val - train_min) / (train_max - train_min)
    X_test = (X_test - train_min) / (train_max - train_min)

else:

    X_train = X_train
    X_val = X_val
    X_test = X_test


# 7. RESHAPE FOR RNN

# RNN input format:
#
# (samples, time_steps, features)
#
# Here:
# time_steps = 187
# features = 1

X_train = X_train.reshape(-1, 187, 1)
X_val = X_val.reshape(-1, 187, 1)
X_test = X_test.reshape(-1, 187, 1)


print("\nAfter reshaping:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# 8. MODEL


model = Sequential()

model.add(Input(shape=(187, 1)))

model.add(
    SimpleRNN(
        64,
        activation="tanh"
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        5,
        activation="softmax"
    )
)


# 9. COMPILE MODEL


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# Display model architecture
print("\nModel Summary:")
model.summary()


# 10. EARLY STOPPING


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# 11. TRAINING


print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=15,
    batch_size=64,

    callbacks=[
        early_stop
    ],

    verbose=1
)



# 12. EVALUATION


print("\nEvaluating model on test data...")

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


# 13. FINAL RESULT

print("RNN TEST RESULTS")
print("Test Loss:", loss)
print("RNN Test Accuracy:", acc)

