import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


class MNISTModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.model_path = 'models/mnist_cnn.keras'
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            
            layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=10, batch_size=128):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        
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
        
        return predicted_class, confidence, predictions[0]
    
    def predict_batch(self, images):
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences, predictions
    
    def get_model_summary(self):
        if self.model is None:
            self.build_model()
        return self.model.summary()
