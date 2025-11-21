import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mnist_model import MNISTModel

print("=" * 60)
print("TRAINING MNIST MODEL")
print("=" * 60)

mnist_model = MNISTModel()
print("\nBuilding MNIST CNN architecture...")
mnist_model.build_model()

print("\nStarting training (10 epochs, batch size 128)...")
mnist_history = mnist_model.train(epochs=10, batch_size=128)

print("\n" + "=" * 60)
print("MNIST TRAINING COMPLETED!")
print(f"Final Training Accuracy: {mnist_history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {mnist_history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Model saved to: {mnist_model.model_path}")
print("=" * 60)
