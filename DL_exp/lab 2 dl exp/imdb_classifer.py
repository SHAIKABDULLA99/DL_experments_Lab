
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save correctly on any machine
                        # (headless servers, scripts run without a display, etc.)
 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
print("=" * 60)
print("IMDB MOVIE REVIEW SENTIMENT CLASSIFICATION")
print("=" * 60)
 
print("TensorFlow Version :", tf.__version__)
 
# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
 
print("\nLoading IMDB Dataset...")
 
vocab_size = 10000
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)
 
print("Training Reviews :", len(x_train))
print("Training Labels  :", len(y_train))
print("Testing Reviews  :", len(x_test))
print("Testing Labels   :", len(y_test))
 
print("\nSample Review (Encoded):")
print(x_train[0][:30])
 
print("\nSample Label :", y_train[0])
 
# --------------------------------------------------
# Pad Sequences
# --------------------------------------------------
 
print("\nPadding Sequences...")
 
max_length = 200
 
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train,
    maxlen=max_length
)
 
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test,
    maxlen=max_length
)
 
print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)
 
# --------------------------------------------------
# Build Model
# --------------------------------------------------
 
print("\nBuilding Neural Network...")
 
model = tf.keras.Sequential([
 
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length
    ),
 
    tf.keras.layers.GlobalAveragePooling1D(),
 
    tf.keras.layers.Dense(128, activation="tanh"),
 
    tf.keras.layers.Dense(64, activation="tanh"),
 
    tf.keras.layers.Dense(1, activation="tanh")
 
])
 
# --------------------------------------------------
# Compile
# --------------------------------------------------
 
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
 
print("\nMODEL SUMMARY\n")
model.summary()
 
# --------------------------------------------------
# Train
# --------------------------------------------------
 
print("\nTraining Model...\n")
 
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
 
# --------------------------------------------------
# Evaluate
# --------------------------------------------------
 
print("\nEvaluating Model...\n")
 
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
 
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
 
# --------------------------------------------------
# Predictions
# --------------------------------------------------
 
print("\nMaking Predictions...\n")
 
predictions = model.predict(x_test[:10])
 
print("=" * 60)
 
for i in range(10):
    prediction = predictions[i][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"
 
    print(f"Review {i+1}")
    print("Predicted :", sentiment)
    print("Actual    :", actual)
    print(f"Probability : {prediction:.4f}")
    print("-" * 40)
 
# --------------------------------------------------
# Accuracy & Loss Graphs (combined figure, saved to disk)
# --------------------------------------------------
 
print("\nGenerating Graphs...\n")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Accuracy subplot
axes[0].plot(history.history["accuracy"], label="Training Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("IMDB Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)
 
# Loss subplot
axes[1].plot(history.history["loss"], label="Training Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("IMDB Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)
 
plt.tight_layout()
 
# Save FIRST (this is the part that actually produces the visible output)
output_path = "imdb_training_graphs.png"
plt.savefig(output_path, dpi=150)
print(f"Graphs saved to: {output_path}")
 
# plt.show() only works if you have a live display (e.g. local Jupyter/desktop).
# It's harmless to leave in, but don't rely on it -- the savefig above is what
# guarantees you actually get the image.
plt.show()
plt.close(fig)
 
print("\nProgram Executed Successfully.")