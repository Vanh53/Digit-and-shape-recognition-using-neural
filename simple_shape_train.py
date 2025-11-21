import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
from shape_generator import ShapeGenerator
import numpy as np

print("Creating and training simplified Shape Model...")

print("Generating dataset (600 samples per class)...")
X, y = ShapeGenerator.generate_dataset(samples_per_class=600, size=64, augment=True)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("Building simplified CNN...")
model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining (12 epochs)...")
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=12,
    validation_data=(X_test, y_test),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ],
    verbose=2
)

acc = history.history['val_accuracy'][-1]
print(f"\nFinal validation accuracy: {acc*100:.1f}%")

os.makedirs('models', exist_ok=True)
model.save('models/shape_cnn_64.keras')
print("Model saved to models/shape_cnn_64.keras")
