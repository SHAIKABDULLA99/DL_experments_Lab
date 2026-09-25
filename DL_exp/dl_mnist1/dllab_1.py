import os

# Reduce TensorFlow startup warnings for cleaner console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = BASE_DIR
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(fig, file_name):
    path = os.path.join(PLOTS_DIR, file_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


print("=" * 50)
print("MLP FOR MNIST HANDWRITTEN DIGIT CLASSIFICATION")
print("=" * 50)

# Optional: make results more consistent across runs
try:
    tf.keras.utils.set_random_seed(42)
except Exception:
    pass

print("TensorFlow Version:", tf.__version__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

# --------------------------------------------------
# Display Sample Images
# --------------------------------------------------

sample_fig = plt.figure(figsize=(10, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.tight_layout()
save_plot(sample_fig, "sample_images.png")

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("\nData preprocessing completed.")

# --------------------------------------------------
# Build Model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="tanh"),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY\n")
model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining Started...\n")

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

print("\nEvaluating...\n")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(x_test)

print("\nDisplaying Sample Predictions...")

prediction_fig = plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    img = x_test[i].reshape(28, 28)
    pred = np.argmax(predictions[i])
    actual = np.argmax(y_test[i])

    plt.imshow(img, cmap="gray")
    plt.title(f"P:{pred}\nA:{actual}")
    plt.axis("off")

plt.tight_layout()
save_plot(prediction_fig, "predictions.png")

# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------

accuracy_fig = plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
save_plot(accuracy_fig, "accuracy.png")

# --------------------------------------------------
# Loss Graph
# --------------------------------------------------

loss_fig = plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
save_plot(loss_fig, "loss.png")

print("\nProgram Executed Successfully.")