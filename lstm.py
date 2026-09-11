import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. LOAD DATA
# -----------------------------
try:
    train_df = pd.read_csv("mitbih_train.csv")
    test_df = pd.read_csv("mitbih_test.csv")
except:
    print("Dataset not found, using dummy data...")
    train_df = pd.DataFrame(np.random.rand(1000, 188))
    test_df = pd.DataFrame(np.random.rand(200, 188))
    train_df[187] = np.random.randint(0, 5, 1000)
    test_df[187] = np.random.randint(0, 5, 200)

# Split features and labels
X = train_df.iloc[:, :-1].values   # 187 features
y = train_df.iloc[:, -1].values    # labels

# -----------------------------
# 2. PREPROCESSING
# -----------------------------

# Normalize (0–1)
X = X / np.max(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Reshape for LSTM → (samples, timesteps, features)
X_train = X_train.reshape(-1, 187, 1)
X_val = X_val.reshape(-1, 187, 1)

# -----------------------------
# 3. MODEL BUILDING (LSTM)
# -----------------------------
model = Sequential()

model.add(LSTM(64, input_shape=(187, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# -----------------------------
# 4. TRAINING
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=15,
          batch_size=64,
          callbacks=[early_stop])

# -----------------------------
# 5. TEST EVALUATION
# -----------------------------
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

X_test = X_test / np.max(X_test)
X_test = X_test.reshape(-1, 187, 1)

loss, acc = model.evaluate(X_test, y_test)

print("LSTM Test Accuracy:", acc)

