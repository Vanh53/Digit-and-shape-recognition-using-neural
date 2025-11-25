import numpy as np
from PIL import Image, ImageDraw
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
    def _create_canvas(size):
        """Hàm phụ trợ tạo nền đen"""
        # Tạo ảnh mode 'L' (Luminance - Grayscale 8-bit), màu đen (0)
        return Image.new('L', (size, size), 0)

    @staticmethod
    def _to_numpy(pil_image):
        """Chuyển từ PIL Image sang Numpy array"""
        return np.array(pil_image, dtype=np.uint8)

    @staticmethod
    def create_circle(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = size // 2
        radius = np.random.randint(size // 4, size // 2 - 5)
        
        # PIL vẽ ellipse/circle qua bounding box (x0, y0, x1, y1)
        x0 = center - radius
        y0 = center - radius
        x1 = center + radius
        y1 = center + radius
        
        draw.ellipse([x0, y0, x1, y1], fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_rectangle(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        width = np.random.randint(size // 3, size - 10)
        height = np.random.randint(size // 4, size - 10)
        x = (size - width) // 2
        y = (size - height) // 2
        
        draw.rectangle([x, y, x + width, y + height], fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_square(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        side = np.random.randint(size // 2, size - 10)
        x = (size - side) // 2
        y = (size - side) // 2
        
        draw.rectangle([x, y, x + side, y + side], fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_triangle(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = size // 2
        height = np.random.randint(size // 2, size - 10)
        base = np.random.randint(size // 2, size - 10)
        
        # Các đỉnh (points)
        p1 = (center, center - height // 2)
        p2 = (center - base // 2, center + height // 2)
        p3 = (center + base // 2, center + height // 2)
        
        draw.polygon([p1, p2, p3], fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_pentagon(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = (size // 2, size // 2)
        radius = np.random.randint(size // 4, size // 2 - 5)
        
        pts = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            pts.append((x, y))
        
        draw.polygon(pts, fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_hexagon(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = (size // 2, size // 2)
        radius = np.random.randint(size // 4, size // 2 - 5)
        
        pts = []
        for i in range(6):
            angle = i * 2 * np.pi / 6
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            pts.append((x, y))
        
        draw.polygon(pts, fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_oval(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = size // 2
        axes_a = np.random.randint(size // 4, size // 2 - 5) # Bán trục lớn/nhỏ
        axes_b = np.random.randint(size // 6, axes_a - 3)
        
        # Bounding box cho oval
        x0 = center - axes_a
        y0 = center - axes_b
        x1 = center + axes_a
        y1 = center + axes_b
        
        draw.ellipse([x0, y0, x1, y1], fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def create_diamond(size=64):
        image = ShapeGenerator._create_canvas(size)
        draw = ImageDraw.Draw(image)
        
        center = size // 2
        half_size = np.random.randint(size // 4, size // 2 - 5)
        
        pts = [
            (center, center - half_size),
            (center + half_size, center),
            (center, center + half_size),
            (center - half_size, center)
        ]
        
        draw.polygon(pts, fill=255)
        return ShapeGenerator._to_numpy(image)
    
    @staticmethod
    def add_noise(image, noise_level=0.1):
        # Noise dùng numpy nên không cần đổi
        noise = np.random.randn(*image.shape) * noise_level * 255
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def rotate_image(image_array, angle=None):
        if angle is None:
            angle = np.random.randint(0, 360)
            
        # Chuyển array sang PIL Image để xoay
        pil_image = Image.fromarray(image_array)
        # resample=Image.BICUBIC giúp ảnh mượt hơn khi xoay
        rotated_pil = pil_image.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        return np.array(rotated_pil)
    
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
                
                # Chuẩn hóa về khoảng 0-1
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
