import numpy as np
import cv2
from PIL import Image
import os


class ShapeGenerator:
    
    SHAPE_CLASSES = {
        0: 'Circle',
        1: 'Rectangle', 
        2: 'Square',
        3: 'Triangle',
        4: 'Pentagon',
        5: 'Hexagon',
        6: 'Oval',
        7: 'Diamond'
    }
    
    @staticmethod
    def create_circle(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        radius = np.random.randint(size // 4, size // 2 - 5)
        cv2.circle(image, (center, center), radius, 255, -1)
        return image
    
    @staticmethod
    def create_rectangle(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        width = np.random.randint(size // 3, size - 10)
        height = np.random.randint(size // 4, size - 10)
        x = (size - width) // 2
        y = (size - height) // 2
        cv2.rectangle(image, (x, y), (x + width, y + height), 255, -1)
        return image
    
    @staticmethod
    def create_square(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        side = np.random.randint(size // 2, size - 10)
        x = (size - side) // 2
        y = (size - side) // 2
        cv2.rectangle(image, (x, y), (x + side, y + side), 255, -1)
        return image
    
    @staticmethod
    def create_triangle(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        height = np.random.randint(size // 2, size - 10)
        base = np.random.randint(size // 2, size - 10)
        
        pts = np.array([
            [center, center - height // 2],
            [center - base // 2, center + height // 2],
            [center + base // 2, center + height // 2]
        ], np.int32)
        
        cv2.fillPoly(image, [pts], 255)
        return image
    
    @staticmethod
    def create_pentagon(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = np.random.randint(size // 4, size // 2 - 5)
        
        pts = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            pts.append([x, y])
        
        pts = np.array(pts, np.int32)
        cv2.fillPoly(image, [pts], 255)
        return image
    
    @staticmethod
    def create_hexagon(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = np.random.randint(size // 4, size // 2 - 5)
        
        pts = []
        for i in range(6):
            angle = i * 2 * np.pi / 6
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            pts.append([x, y])
        
        pts = np.array(pts, np.int32)
        cv2.fillPoly(image, [pts], 255)
        return image
    
    @staticmethod
    def create_oval(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        axes_a = np.random.randint(size // 4, size // 2 - 5)
        axes_b = np.random.randint(size // 6, axes_a - 3)
        cv2.ellipse(image, center, (axes_a, axes_b), 0, 0, 360, 255, -1)
        return image
    
    @staticmethod
    def create_diamond(size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        half_size = np.random.randint(size // 4, size // 2 - 5)
        
        pts = np.array([
            [center, center - half_size],
            [center + half_size, center],
            [center, center + half_size],
            [center - half_size, center]
        ], np.int32)
        
        cv2.fillPoly(image, [pts], 255)
        return image
    
    @staticmethod
    def add_noise(image, noise_level=0.1):
        noise = np.random.randn(*image.shape) * noise_level * 255
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def rotate_image(image, angle=None):
        if angle is None:
            angle = np.random.randint(0, 360)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=0)
        return rotated
    
    @staticmethod
    def generate_dataset(samples_per_class=1000, size=64, augment=True):
        shape_functions = [
            ShapeGenerator.create_circle,
            ShapeGenerator.create_rectangle,
            ShapeGenerator.create_square,
            ShapeGenerator.create_triangle,
            ShapeGenerator.create_pentagon,
            ShapeGenerator.create_hexagon,
            ShapeGenerator.create_oval,
            ShapeGenerator.create_diamond
        ]
        
        X = []
        y = []
        
        for class_idx, shape_func in enumerate(shape_functions):
            for _ in range(samples_per_class):
                image = shape_func(size)
                
                if augment and np.random.random() > 0.5:
                    image = ShapeGenerator.rotate_image(image)
                
                if augment and np.random.random() > 0.7:
                    image = ShapeGenerator.add_noise(image, noise_level=0.05)
                
                image = image.astype(np.float32) / 255.0
                image = image.reshape(size, size, 1)
                
                X.append(image)
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def save_sample_images(output_dir='sample_images/shapes', samples=5):
        os.makedirs(output_dir, exist_ok=True)
        
        shape_functions = [
            (ShapeGenerator.create_circle, 'circle'),
            (ShapeGenerator.create_rectangle, 'rectangle'),
            (ShapeGenerator.create_square, 'square'),
            (ShapeGenerator.create_triangle, 'triangle'),
            (ShapeGenerator.create_pentagon, 'pentagon'),
            (ShapeGenerator.create_hexagon, 'hexagon'),
            (ShapeGenerator.create_oval, 'oval'),
            (ShapeGenerator.create_diamond, 'diamond')
        ]
        
        for shape_func, name in shape_functions:
            for i in range(samples):
                image = shape_func(128)
                image_pil = Image.fromarray(image)
                image_pil.save(f"{output_dir}/{name}_{i+1}.png")
