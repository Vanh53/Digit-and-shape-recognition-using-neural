import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mnist_model import MNISTModel
from shape_model import ShapeModel
from shape_generator import ShapeGenerator

print("=" * 60)
print("TRAINING MODELS FOR CNN IMAGE RECOGNITION PROJECT")
print("=" * 60)

print("\n" + "=" * 60)
print("STEP 1: Training MNIST Model")
print("=" * 60)
mnist_model = MNISTModel()
print("\nBuilding MNIST CNN architecture...")
mnist_model.build_model()
print("\nModel Summary:")
mnist_model.model.summary()

print("\nStarting training (10 epochs)...")
mnist_history = mnist_model.train(epochs=10, batch_size=128)

print("\n" + "=" * 60)
print("MNIST TRAINING COMPLETED!")
print(f"Final Training Accuracy: {mnist_history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {mnist_history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Model saved to: {mnist_model.model_path}")
print("=" * 60)

print("\n" + "=" * 60)
print("STEP 2: Training Shape Recognition Model")
print("=" * 60)
shape_model = ShapeModel(input_size=64)
print("\nBuilding Shape CNN architecture...")
shape_model.build_model()
print("\nModel Summary:")
shape_model.model.summary()

print("\nStarting training (30 epochs, 1000 samples per class)...")
shape_history = shape_model.train(epochs=30, batch_size=64, samples_per_class=1000)

print("\n" + "=" * 60)
print("SHAPE MODEL TRAINING COMPLETED!")
print(f"Final Training Accuracy: {shape_history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {shape_history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Model saved to: {shape_model.model_path}")
print("=" * 60)

print("\n" + "=" * 60)
print("STEP 3: Generating Sample Images")
print("=" * 60)
print("Creating sample images for demo...")
ShapeGenerator.save_sample_images(samples=5)
print("Sample images created in sample_images/shapes/")

print("\n" + "=" * 60)
print("ALL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nModels are ready to use:")
print(f"  ✓ MNIST Model: {mnist_model.model_path}")
print(f"  ✓ Shape Model: {shape_model.model_path}")
print(f"\nMNIST Validation Accuracy: {mnist_history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Shape Validation Accuracy: {shape_history.history['val_accuracy'][-1]*100:.2f}%")
print("\nYou can now run the Streamlit app!")
print("=" * 60)
