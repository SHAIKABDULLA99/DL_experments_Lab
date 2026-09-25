# reuters_multiclass_classification.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("REUTERS NEWSWIRE MULTI-CLASS CLASSIFICATION")
print("=" * 60)
print("TensorFlow Version :", tf.__version__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
print("\nLoading Reuters Dataset...")
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(
    num_words=vocab_size
)

print("Training News Articles :", len(x_train))
print("Training Labels        :", len(y_train))
print("Testing News Articles  :", len(x_test))
print("Testing Labels         :", len(y_test))
print("\nNumber of Classes :", np.max(y_train) + 1)
print("\nSample Encoded Article:")
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
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(46, activation="softmax")
])

# --------------------------------------------------
# Compile Model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
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
    epochs=10,
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
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
print("\nMaking Predictions...\n")
predictions = model.predict(x_test[:10])
print("=" * 60)
for i in range(10):
    predicted_class = np.argmax(predictions[i])
    actual_class = y_test[i]
    print(f"Article {i+1}")
    print("Predicted Class :", predicted_class)
    print("Actual Class    :", actual_class)
    print("-" * 40)

# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Reuters Dataset Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# Loss Graph
# --------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Reuters Dataset Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("\nProgram Executed Successfully.")