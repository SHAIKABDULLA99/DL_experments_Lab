# EXP 6: CNN for Dogs and Cats Classification

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 60)
print("CNN FOR DOGS AND CATS IMAGE CLASSIFICATION")
print("=" * 60)

print("TensorFlow Version:", tf.__version__)

# --------------------------------------------------
# Dataset Path
# --------------------------------------------------

dataset_path = Path(__file__).resolve().parent / "dogs_cats"
output_path = Path(__file__).resolve().parent

# Folder structure:
# dogs_cats/
#     cats/
#         cat1.jpg
#         cat2.jpg
#     dogs/
#         dog1.jpg
#         dog2.jpg

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

img_height = 150
img_width = 150
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_dataset.class_names

print("\nClasses:", class_names)

# --------------------------------------------------
# Display Sample Images
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for images, labels in train_dataset.take(1):

    sample_count = min(9, images.shape[0])

    for i in range(sample_count):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.tight_layout()
plt.savefig(output_path / "training_samples.png")
plt.close()

# --------------------------------------------------
# Build CNN Model
# --------------------------------------------------

print("\nBuilding CNN Model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(img_height, img_width, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(
        (2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(
        (2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(
        (2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"),

    tf.keras.layers.Dense(
        1, activation="sigmoid")
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

print("\nTraining CNN...\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10,

    shuffle=False
)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    validation_dataset
)

print("Validation Loss     :", loss)

print("Validation Accuracy :", accuracy)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

print("\nMaking Prediction...")

for images, labels in validation_dataset.take(1):

    predictions = model.predict(images)

    plt.figure(figsize=(10, 6))

    sample_count = min(9, images.shape[0])

    for i in range(sample_count):

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        predicted = (
            "Dog" if predictions[i][0] >= 0.5
            else "Cat"
        )

        actual = class_names[labels[i]]

        plt.title(
            f"Predicted: {predicted}\nActual: {actual}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_path / "predictions.png")
    plt.close()

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

plt.title("CNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(output_path / "accuracy.png")
plt.close()

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

plt.title("CNN Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(output_path / "loss.png")
plt.close()

print("\nProgram Executed Successfully.")