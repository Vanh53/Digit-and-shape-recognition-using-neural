import cv2
import numpy as np
from PIL import Image


class MultiObjectDetector:
    
    def __init__(self, model, preprocessor, min_area=100):
        self.model = model
        self.preprocessor = preprocessor
        self.min_area = min_area
    
    def detect_objects(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(gray.shape[1], x + w + margin)
            y2 = min(gray.shape[0], y + h + margin)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            preprocessed = self.preprocessor(roi)
            
            try:
                if hasattr(self.model, 'predict'):
                    result = self.model.predict(preprocessed)
                    
                    if len(result) == 4:
                        pred_class, shape_name, confidence, probs = result
                    elif len(result) == 3:
                        pred_class, confidence, probs = result
                        shape_name = str(pred_class)
                    else:
                        pred_class = result[0]
                        confidence = 0.0
                        shape_name = str(pred_class)
                    
                    detected_objects.append({
                        'bbox': (x, y, w, h),
                        'class': pred_class,
                        'name': shape_name,
                        'confidence': float(confidence),
                        'area': area
                    })
            except Exception as e:
                print(f"Error predicting object: {e}")
                continue
        
        detected_objects.sort(key=lambda x: x['area'], reverse=True)
        
        return detected_objects
    
    def draw_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            result = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{detection['name']}: {detection['confidence']:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        return result
    
    def get_detection_summary(self, detections):
        summary = {
            'total_objects': len(detections),
            'objects_by_class': {}
        }
        
        for detection in detections:
            name = detection['name']
            if name not in summary['objects_by_class']:
                summary['objects_by_class'][name] = 0
            summary['objects_by_class'][name] += 1
        
        return summary
