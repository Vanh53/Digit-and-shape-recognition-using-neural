import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from shape_model import ShapeModel

print("Training Shape Model only...")
shape_model = ShapeModel(input_size=64)
shape_model.build_model()

history = shape_model.train(epochs=20, batch_size=128, samples_per_class=800)
acc = history.history['val_accuracy'][-1]
print(f"\n✓ Shape Model complete: {acc*100:.1f}% validation accuracy")
print(f"Model saved to: {shape_model.model_path}")
