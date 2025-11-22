import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    
    @staticmethod
    def resize_image(image, size=(28, 28)):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image):
        image = image.astype(np.float32)
        return image / 255.0
    
    @staticmethod
    def convert_to_grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    @staticmethod
    def preprocess_for_mnist(image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = ImageProcessor.convert_to_grayscale(image)
        
        resized = ImageProcessor.resize_image(gray, (28, 28))
        
        normalized = ImageProcessor.normalize_image(resized)
        
        reshaped = normalized.reshape(28, 28, 1)
        
        return reshaped
    
    @staticmethod
    def preprocess_for_shapes(image, target_size=(64, 64)):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = ImageProcessor.convert_to_grayscale(image)
        
        resized = ImageProcessor.resize_image(gray, target_size)
        
        normalized = ImageProcessor.normalize_image(resized)
        
        reshaped = normalized.reshape(target_size[0], target_size[1], 1)
        
        return reshaped
