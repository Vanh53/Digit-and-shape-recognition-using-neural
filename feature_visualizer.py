import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2


class FeatureVisualizer:
    
    def __init__(self, model):
        self.model = model
        self.layer_outputs = None
        self.activation_model = None
    
    def get_layer_names(self):
        return [layer.name for layer in self.model.layers]
    
    def get_conv_layers(self):
        conv_layers = []
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
        return conv_layers
    
    def build_activation_model(self, layer_names=None):
        if layer_names is None:
            layer_names = self.get_conv_layers()
        
        layer_outputs = []
        for name in layer_names:
            try:
                layer = self.model.get_layer(name)
                layer_outputs.append(layer.output)
            except:
                continue
        
        if layer_outputs:
            self.activation_model = keras.Model(
                inputs=self.model.input,
                outputs=layer_outputs
            )
            self.layer_outputs = layer_outputs
            return True
        return False
    
    def get_activations(self, image):
        if self.activation_model is None:
            self.build_activation_model()
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        activations = self.activation_model.predict(image, verbose=0)
        
        if not isinstance(activations, list):
            activations = [activations]
        
        return activations
    
    def visualize_feature_maps(self, image, layer_index=0, max_filters=16):
        activations = self.get_activations(image)
        
        if layer_index >= len(activations):
            layer_index = 0
        
        layer_activation = activations[layer_index]
        
        n_filters = min(layer_activation.shape[-1], max_filters)
        
        n_cols = 4
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
        axes = axes.flatten() if n_filters > 1 else [axes]
        
        conv_layers = self.get_conv_layers()
        layer_name = conv_layers[layer_index] if layer_index < len(conv_layers) else f"Layer {layer_index}"
        
        fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16)
        
        for i in range(n_filters):
            feature_map = layer_activation[0, :, :, i]
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        for i in range(n_filters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_all_layers(self, image, max_filters_per_layer=8):
        activations = self.get_activations(image)
        conv_layers = self.get_conv_layers()
        
        figures = []
        
        for idx, (activation, layer_name) in enumerate(zip(activations, conv_layers)):
            n_filters = min(activation.shape[-1], max_filters_per_layer)
            
            n_cols = 4
            n_rows = (n_filters + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2.5))
            axes = axes.flatten() if n_filters > 1 else [axes]
            
            fig.suptitle(f'Layer: {layer_name} - Shape: {activation.shape}', fontsize=14)
            
            for i in range(n_filters):
                feature_map = activation[0, :, :, i]
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Filter {i}', fontsize=10)
                axes[i].axis('off')
            
            for i in range(n_filters, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            figures.append(fig)
        
        return figures
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_confusion_matrix_plot(y_true, y_pred, class_names):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        return fig
