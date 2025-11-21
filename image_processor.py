import cv2
import numpy as np
from skimage import filters, morphology, segmentation
from skimage.filters import threshold_otsu
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
    def apply_gaussian_blur(image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def apply_median_filter(image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def sharpen_image(image):
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def detect_edges_canny(image, threshold1=100, threshold2=200):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, threshold1, threshold2)
    
    @staticmethod
    def detect_edges_sobel(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        return sobel
    
    @staticmethod
    def threshold_binary(image, threshold_value=127):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def threshold_otsu(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        threshold_value = threshold_otsu(gray)
        binary = gray > threshold_value
        return (binary * 255).astype(np.uint8)
    
    @staticmethod
    def adaptive_threshold(image, block_size=11, C=2):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, C)
    
    @staticmethod
    def morphological_operations(image, operation='close', kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'erode':
            return cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image
    
    @staticmethod
    def find_contours(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def watershed_segmentation(image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(image, markers)
        
        segmented = image.copy()
        segmented[markers == -1] = [255, 0, 0]
        
        return segmented
    
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
