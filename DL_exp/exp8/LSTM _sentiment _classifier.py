# Ex8: LSTM Sentiment Classifier
# LSTM for IMDB Sentiment Analysis

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("LSTM FOR IMDB SENTIMENT ANALYSIS")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading IMDB Dataset...")

VOCAB_SIZE = 10000
MAX_LENGTH = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Training Samples :", len(x_train))
print("Testing Samples :", len(x_test))

# --------------------------------------------------
# Pad Sequences
# --------------------------------------------------

x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train,
    maxlen=MAX_LENGTH
)

x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test,
    maxlen=MAX_LENGTH
)

print("Training Shape :", x_train.shape)
print("Testing Shape :", x_test.shape)

# --------------------------------------------------
# Build LSTM Model
# --------------------------------------------------

print("\nBuilding LSTM Model...")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        input_length=MAX_LENGTH
    ),

    tf.keras.layers.LSTM(128),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY\n")
model.summary()

# --------------------------------------------------
# Train Model
# --------------------------------------------------

print("\nTraining Model...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Loss : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(
    x_test[:10],
    verbose=0
)

print("\nSample Predictions\n")

for i in range(10):

    predicted = (
        "Positive"
        if predictions[i][0] > 0.5
        else "Negative"
    )

    actual = (
        "Positive"
        if y_test[i] == 1
        else "Negative"
    )

    print(f"Review {i+1}")
    print("Predicted :", predicted)
    print("Actual :", actual)
    print("-" * 40)

# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# Loss Graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("\nProgram Executed Successfully.")