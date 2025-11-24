import numpy as np
from PIL import Image


class ImageProcessor:
    
    @staticmethod
    def resize_image(image, size=(28, 28)):
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Lấy kích thước hiện tại
        old_height, old_width = image.shape[:2]
        new_height, new_width = size
        
        # Tính tỷ lệ scale
        row_ratio = old_height / new_height
        col_ratio = old_width / new_width
        
        # Tạo mảng indices cho nearest neighbor
        row_idx = np.floor(np.arange(new_height) * row_ratio).astype(int)
        col_idx = np.floor(np.arange(new_width) * col_ratio).astype(int)
        
        # Đảm bảo không vượt quá bounds
        row_idx = np.clip(row_idx, 0, old_height - 1)
        col_idx = np.clip(col_idx, 0, old_width - 1)
        
        # Resize bằng nearest neighbor
        if len(image.shape) == 2:  # Grayscale
            resized = image[row_idx[:, None], col_idx]
        else:  # RGB
            resized = image[row_idx[:, None], col_idx]
        
        return resized
    
    @staticmethod
    def normalize_image(image):
        """
        Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1]
        """
        image = image.astype(np.float32)
        return image / 255.0
    
    @staticmethod
    def convert_to_grayscale(image):
        """
        Chuyển ảnh RGB sang grayscale bằng công thức chuẩn
        Gray = 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) == 3:
            # Lấy 3 channels đầu tiên (RGB hoặc RGBA)
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            
            # Áp dụng công thức grayscale chuẩn
            gray = (0.299 * image[:, :, 0] + 
                   0.587 * image[:, :, 1] + 
                   0.114 * image[:, :, 2])
            
            return gray.astype(np.uint8)
        
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
