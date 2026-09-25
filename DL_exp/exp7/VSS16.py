# =================================================
# Experiment 7
# Transfer Learning using VGG16 (FAST VERSION)
# =================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRANSFER LEARNING USING VGG16 (FAST VERSION)")
print("=" * 60)
print("TensorFlow Version :", tf.__version__)

# -------------------------------------------------
# Download Dataset
# -------------------------------------------------
print("\nDownloading Flower Dataset...")
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_path = tf.keras.utils.get_file(
    "flower_photos",
    origin=dataset_url,
    untar=True
)
print("Dataset Path :", dataset_path)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("\nClasses :", class_names)

# -------------------------------------------------
# Show Sample Images
# -------------------------------------------------
plt.figure(figsize=(8, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# FAST MODE
# -------------------------------------------------
train_ds = train_ds.take(20)
val_ds = val_ds.take(5)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# -------------------------------------------------
# Load VGG16
# -------------------------------------------------
print("\nLoading Pretrained VGG16...")
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)
base_model.trainable = False

# -------------------------------------------------
# Build Model
# -------------------------------------------------
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# -------------------------------------------------
# Compile Model
# -------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY\n")
model.summary()

# -------------------------------------------------
# Train Model
# -------------------------------------------------
print("\nTraining Model...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1
)

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------
print("\nEvaluating Model...\n")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# -------------------------------------------------
# Sample Predictions
# -------------------------------------------------
print("\nSample Predictions\n")
for images, labels in val_ds.take(1):
    predictions = model.predict(images[:5], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    for i in range(5):
        print(f"Image {i+1}")
        print("Predicted :", class_names[predicted_classes[i]])
        print("Actual    :", class_names[labels[i]])
        print("-" * 40)

# -------------------------------------------------
# Accuracy Graph
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------
# Loss Graph
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("\nProgram Executed Successfully.")