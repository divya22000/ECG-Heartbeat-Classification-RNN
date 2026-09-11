
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Keep the results similar each time we run the program
np.random.seed(42)
tf.random.set_seed(42)


# Small dataset for training the sentiment model
texts = [
    "i love this movie",
    "this is amazing",
    "i like it",
    "so good",
    "i hate this",
    "this is bad",
    "terrible experience",
    "not good"
]

# 1 means positive and 0 means negative
labels = [
    1, 1, 1, 1,
    0, 0, 0, 0
]


# Add the examples a few times to give the model more samples
texts = texts * 20
labels = labels * 20

texts = np.array(texts)
labels = np.array(labels)


# Shuffle the data before splitting it
indices = np.arange(len(texts))
np.random.shuffle(indices)

texts = texts[indices]
labels = labels[indices]


# Keep separate data for training, validation and testing
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

print("Training samples:", len(train_texts))
print("Validation samples:", len(val_texts))
print("Test samples:", len(test_texts))


# Convert sentences into numbers
vocab_size = 1000
max_len = 10

vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len
)

# Create the vocabulary using the training sentences
vectorizer.adapt(train_texts)

X_train = vectorizer(train_texts)
X_val = vectorizer(val_texts)
X_test = vectorizer(test_texts)

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)


# Add word meaning and word position information
class PositionalEmbedding(layers.Layer):

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.pos_emb = layers.Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

    def call(self, x):

        positions = tf.range(
            start=0,
            limit=tf.shape(x)[-1],
            delta=1
        )

        token_embedding = self.token_emb(x)
        position_embedding = self.pos_emb(positions)

        return token_embedding + position_embedding


# This block helps the model understand relationships between words
class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.feed_forward = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation="relu"
            ),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):

        attention_output = self.attention(
            inputs,
            inputs,
            training=training
        )

        x = self.norm1(
            inputs + attention_output
        )

        feed_forward_output = self.feed_forward(
            x,
            training=training
        )

        return self.norm2(
            x + feed_forward_output
        )


# Transformer settings
embed_dim = 32
num_heads = 2
ff_dim = 32


# Build the model
inputs = layers.Input(
    shape=(max_len,)
)

x = PositionalEmbedding(
    max_len,
    vocab_size,
    embed_dim
)(inputs)

x = TransformerBlock(
    embed_dim,
    num_heads,
    ff_dim
)(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(
    32,
    activation="relu"
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)


# Prepare the model for positive/negative classification
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    shuffle=True,
    verbose=1
)


# Check the model using the test data
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Accuracy:", test_accuracy)


# Show training and validation accuracy
plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer Model Accuracy")
plt.legend()

plt.show()


# Show training and validation loss
plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Model Loss")
plt.legend()

plt.show()


# Let the user enter a message and get its sentiment

print("Sentiment Prediction")
print("Type 'exit' when you want to stop.\n")


while True:

    message = input("Enter your message: ")

    # Stop the program
    if message.lower() == "exit":
        print("Program stopped.")
        break

    # Check that something was entered
    if message.strip() == "":
        print("Please enter a message.\n")
        continue

    # Convert the message into the format used by the model
    message_sequence = vectorizer(
        np.array([message])
    )

    # Get the model prediction
    prediction = model.predict(
        message_sequence,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    # Convert the probability into a sentiment
    if probability >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print(
        f"Prediction: {sentiment}"
    )

    print(
        f"Confidence: {probability:.2f}"
    )

    print()
