import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("BOSTON HOUSING PRICE PREDICTION")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading Boston Housing Dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

print("Training Samples :", x_train.shape)
print("Training Labels  :", y_train.shape)

print("Testing Samples  :", x_test.shape)
print("Testing Labels   :", y_test.shape)

print("\nNumber of Features :", x_train.shape[1])

# --------------------------------------------------
# Normalize Data
# --------------------------------------------------

print("\nNormalizing Data...")

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

print("Normalization Completed.")

# --------------------------------------------------
# Build Model
# --------------------------------------------------

print("\nBuilding Neural Network...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(13,)),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(1)

])

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

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

    epochs=100,

    batch_size=16,

    validation_split=0.2,

    verbose=1

)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

print("\nEvaluating Model...\n")

loss, mae = model.evaluate(

    x_test,

    y_test,

    verbose=0

)

print(f"Test Loss (MSE) : {loss:.4f}")
print(f"Mean Absolute Error : {mae:.4f}")

# --------------------------------------------------
# Predict House Prices
# --------------------------------------------------

print("\nPredicting House Prices...\n")

predictions = model.predict(x_test[:10])

print("=" * 60)

for i in range(10):

    print(f"House {i+1}")

    print(f"Actual Price    : ${y_test[i]:.2f} Thousand")

    print(f"Predicted Price : ${predictions[i][0]:.2f} Thousand")

    print("-" * 40)

# --------------------------------------------------
# Plot MAE
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["mae"], label="Training MAE")

plt.plot(history.history["val_mae"], label="Validation MAE")

plt.title("Mean Absolute Error")

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()

plt.grid(True)

plt.show()

# --------------------------------------------------
# Plot Loss
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Mean Squared Error")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("\nProgram Executed Successfully.")
