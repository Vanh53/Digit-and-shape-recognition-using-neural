import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from shape_generator import ShapeGenerator


class ShapeModel:
    def __init__(self, input_size=64):
        self.model = None
        self.history = None
        self.input_size = input_size
        self.model_path = f'models/shape_cnn_{input_size}.keras'
        self.shape_classes = ShapeGenerator.SHAPE_CLASSES
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_size, self.input_size, 1)),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.shape_classes), activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=30, batch_size=64, samples_per_class=1000):
        print("Generating shape dataset...")
        X, y = ShapeGenerator.generate_dataset(
            samples_per_class=samples_per_class,
            size=self.input_size,
            augment=True
        )
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        return self.history
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            return True
        return False
    
    def predict(self, image):
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        shape_name = self.shape_classes[predicted_class]
        
        return predicted_class, shape_name, confidence, predictions[0]
    
    def predict_batch(self, images):
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        shape_names = [self.shape_classes[cls] for cls in predicted_classes]
        
        return predicted_classes, shape_names, confidences, predictions
    
    def get_model_summary(self):
        if self.model is None:
            self.build_model()
        return self.model.summary()
