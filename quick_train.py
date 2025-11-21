import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mnist_model import MNISTModel
from shape_model import ShapeModel
from shape_generator import ShapeGenerator
import sys

print("Quick training for pre-deployment...")

print("\n1. Training MNIST (5 epochs for quick setup)...")
mnist_model = MNISTModel()
mnist_model.build_model()
try:
    history = mnist_model.train(epochs=5, batch_size=256)
    acc = history.history['val_accuracy'][-1]
    print(f"✓ MNIST done: {acc*100:.1f}% accuracy")
except Exception as e:
    print(f"✗ MNIST failed: {e}")
    sys.exit(1)

print("\n2. Training Shape Model (15 epochs, 800 samples)...")
shape_model = ShapeModel(input_size=64)
shape_model.build_model()
try:
    history = shape_model.train(epochs=15, batch_size=128, samples_per_class=800)
    acc = history.history['val_accuracy'][-1]
    print(f"✓ Shape done: {acc*100:.1f}% accuracy")
except Exception as e:
    print(f"✗ Shape failed: {e}")
    sys.exit(1)

print("\n3. Creating sample images...")
ShapeGenerator.save_sample_images(samples=5)
print("✓ Samples created")

print("\n=== Training Complete ===")
print("Models are ready to use!")
