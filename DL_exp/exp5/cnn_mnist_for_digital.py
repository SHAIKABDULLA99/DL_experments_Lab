# EXP 5
# Build a Convolutional Neural Network (CNN)
# for MNIST Handwritten Digit Classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("CNN FOR MNIST HANDWRITTEN DIGIT CLASSIFICATION")
print("=" * 60)

print("TensorFlow Version:", tf.__version__)

# --------------------------------------------------
# 1. Load MNIST Dataset
# --------------------------------------------------

print("\nLoading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

# --------------------------------------------------
# 2. Display Sample Images
# --------------------------------------------------

plt.figure(figsize=(10, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

print("\nPreprocessing Data...")

# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
# 28 x 28 -> 28 x 28 x 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("Training Shape:", x_train.shape)
print("Testing Shape :", x_test.shape)

# --------------------------------------------------
# 4. Build CNN Model
# --------------------------------------------------

print("\nBuilding CNN Model...")

model = tf.keras.Sequential([
    
    # Input Layer
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First Convolution Layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="tanh"
    ),

    # First Max Pooling
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Second Max Pooling
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into vector
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output Layer
    # 10 classes: digits 0-9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

# --------------------------------------------------
# 5. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY\n")
model.summary()

# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------

print("\nTraining CNN...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# --------------------------------------------------
# 7. Evaluate Model
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
# 8. Make Predictions
# --------------------------------------------------

print("\nMaking Predictions...")

predictions = model.predict(x_test[:10])

print("\nSample Predictions:")

for i in range(10):
    predicted = np.argmax(predictions[i])
    actual = np.argmax(y_test[i])

    print(
        f"Image {i + 1}: "
        f"Predicted = {predicted}, "
        f"Actual = {actual}"
    )

# --------------------------------------------------
# 9. Display Predictions
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    predicted = np.argmax(predictions[i])
    actual = np.argmax(y_test[i])

    plt.title(
        f"P: {predicted}\nA: {actual}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. Accuracy Graph
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

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# 11. Loss Graph
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

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

print("\nProgram Executed Successfully.")