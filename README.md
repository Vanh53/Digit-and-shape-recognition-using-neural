# 🔍 Nhận Dạng Chữ Viết Tay & Hình Học - CNN

Dự án nhận dạng chữ viết tay và hình học cơ bản sử dụng Mạng Neural Tích Chập (Convolutional Neural Networks - CNN) với xử lý ảnh nâng cao.

## 📋 Mục Lục
- [Tổng Quan](#tổng-quan)
- [Tính Năng](#tính-năng)
- [Công Nghệ](#công-nghệ)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Kiến Trúc CNN](#kiến-trúc-cnn)
- [Xử Lý Ảnh](#xử-lý-ảnh)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)

## 🎯 Tổng Quan

Dự án này là một ứng dụng web giáo dục về Computer Vision và Deep Learning, cho phép bạn:
1. **Tự tay train các mô hình CNN** để hiểu quá trình học của mạng neural
2. **Nhận dạng chữ số viết tay (0-9)** với độ chính xác >95% trên MNIST dataset
3. **Nhận dạng 8+ hình học cơ bản**: Tròn, Chữ nhật, Vuông, Tam giác, Ngũ giác, Lục giác, Oval, Hình thoi
4. **Phát hiện nhiều đối tượng** trong cùng một ảnh với bounding boxes
5. **Khám phá xử lý ảnh**: Filters, Edge Detection, Segmentation
6. **Visualize quá trình học**: Feature maps từ các convolutional layers
> **🎓 Tính chất giáo dục:** Bạn sẽ tự train models để thấy được quá trình CNN học như thế nào. Điều này giúp hiểu sâu hơn về Deep Learning so với việc chỉ sử dụng models có sẵn.

## ✨ Tính Năng

### 1. Nhận Dạng Chữ Số (MNIST)
- ✏️ Vẽ tay trực tiếp trên canvas
- 📤 Upload ảnh chứa chữ số
- 📦 Batch processing - xử lý nhiều ảnh cùng lúc
- 📊 Hiển thị confidence score và phân bố xác suất
- 💾 Export kết quả (CSV/JSON)

### 2. Nhận Dạng Hình Học
- 🔷 Nhận dạng 8 loại hình khác nhau
- ✏️ Vẽ hình trực tiếp hoặc upload ảnh
- 🎨 Demo với ảnh mẫu có sẵn
- 📊 Hiển thị confidence score cho từng dự đoán
- 🎯 Top-K predictions

### 3. Phát Hiện Nhiều Đối Tượng
- 🎯 Tự động phát hiện nhiều hình trong một ảnh
- 📦 Vẽ bounding boxes cho mỗi đối tượng
- 📊 Thống kê số lượng theo loại hình
- 💾 Export chi tiết vị trí và kết quả nhận dạng

### 4. Xử Lý Ảnh Nâng Cao
- **Filters**: Gaussian Blur, Median Filter, Bilateral Filter, Sharpening
- **Edge Detection**: Canny, Sobel
- **Segmentation**: Binary Threshold, Otsu, Adaptive Threshold, Watershed
- 🎛️ Điều chỉnh tham số real-time

### 5. Feature Maps Visualization
- 🔬 Visualize feature maps từ các convolutional layers
- 📊 Hiểu cách CNN trích xuất đặc trưng từ ảnh
- 📈 Theo dõi quá trình training (Loss, Accuracy curves)
- 🎯 Confusion Matrix

## 🛠️ Công Nghệ

### Deep Learning & AI
- **TensorFlow/Keras**: Xây dựng và train CNN models
- **NumPy**: Xử lý ma trận và mảng số

### Computer Vision
- **OpenCV**: Xử lý ảnh, edge detection, contours
- **scikit-image**: Advanced image processing
- **Pillow**: Image I/O operations

### Data Science & Visualization
- **Matplotlib**: Visualize feature maps, training curves
- **Seaborn**: Statistical visualization (confusion matrix)
- **Pandas**: Data manipulation và export
- **scikit-learn**: Metrics và evaluation

### Web Framework
- **Streamlit**: Giao diện web tương tác
- **streamlit-drawable-canvas**: Vẽ tay trực tiếp

## 📥 Cài Đặt

### Yêu Cầu
- Python 3.11+
- pip hoặc uv package manager

### Cài Đặt Dependencies

Tất cả dependencies đã được cài đặt sẵn trong môi trường Replit. Nếu chạy local:

```bash
pip install tensorflow opencv-python streamlit streamlit-drawable-canvas
pip install matplotlib seaborn pandas pillow scikit-learn scikit-image numpy
```

## 🚀 Hướng Dẫn Sử Dụng

> **✅ SẴN SÀNG SỬ DỤNG:** Ứng dụng đã có **pre-trained models** sẵn sàng! Bạn có thể bắt đầu sử dụng các tính năng nhận dạng ngay lập tức mà không cần train.
### Bước 1: Sử Dụng Ngay (Models đã có sẵn)
1. **Khởi động ứng dụng Streamlit**
   - Ứng dụng đã có sẵn **pre-trained models**
    - Trang chủ sẽ hiển thị ✅ models đã sẵn sàng
   
2. **Bắt đầu sử dụng các tính năng:**
   - 🔢 **MNIST**: Nhận dạng chữ số viết tay (99.2% accuracy)
   - 🔷 **Shape**: Nhận dạng hình học (92.5% accuracy)
   - 🎯 **Multi-Object**: Phát hiện nhiều đối tượng
   - 🖼️ **Image Processing**: Xử lý ảnh nâng cao
   - 📊 **Feature Maps**: Visualize CNN layers
> **💡 Models có sẵn:**
> - `models/mnist_cnn.keras` - MNIST (99.2% val accuracy)
> - `models/shape_cnn_64.keras` - Shapes (92.5% val accuracy)
> - `sample_images/shapes/` - 24 ảnh mẫu demo
### Bước 2: (Tùy Chọn) Retrain Models
Nếu bạn muốn **cải thiện độ chính xác** hoặc **học cách train CNN**:
1. Vào trang **"⚙️ Train Model"**
2. **Retrain MNIST Model** (~2-3 phút):
   - Tăng epochs để cải thiện accuracy
   - Thử các hyperparameters khác
3. **Retrain Shape Model** (~5-7 phút):
   - Tăng samples_per_class cho accuracy tốt hơn
   - Tăng epochs (khuyến nghị 30-50 cho >95% accuracy)
4. **Tạo thêm ảnh mẫu** nếu cần
> **🎓 Giá trị giáo dục:** Retrain để thấy ảnh hưởng của hyperparameters, data augmentation, và epochs lên model performance!

### Bước 3: Khám Phá Các Tính Năng

#### 🔢 Nhận Dạng MNIST
1. Vào trang "Nhận Dạng Chữ Số (MNIST)"
2. Chọn tab:
   - **Vẽ Tay**: Vẽ số từ 0-9 trên canvas đen
   - **Upload**: Upload ảnh chứa chữ số
   - **Batch**: Upload nhiều ảnh để xử lý cùng lúc
3. Click "Nhận Dạng"
4. Xem kết quả và confidence score

#### 🔷 Nhận Dạng Hình Học
1. Vào trang "Nhận Dạng Hình Học"
2. Chọn tab:
   - **Vẽ Hình**: Vẽ hình học trên canvas
   - **Upload**: Upload ảnh chứa hình
   - **Demo**: Xem demo với ảnh mẫu
3. Click "Nhận Dạng Hình"
4. Xem kết quả phân loại

#### 🎯 Phát Hiện Nhiều Đối Tượng
1. Vào trang "Phát Hiện Nhiều Đối Tượng"
2. Upload ảnh chứa nhiều hình
3. Hệ thống tự động:
   - Phát hiện các đối tượng
   - Vẽ bounding boxes
   - Hiển thị loại hình và confidence
4. Xem thống kê và export kết quả

#### 🖼️ Xử Lý Ảnh
1. Vào trang "Xử Lý Ảnh Nâng Cao"
2. Upload ảnh
3. Chọn tab:
   - **Filters**: Thử các bộ lọc khác nhau
   - **Edge Detection**: Phát hiện biên cạnh
   - **Segmentation**: Phân đoạn ảnh
4. Điều chỉnh tham số và xem kết quả real-time

#### 📊 Feature Maps
1. Vào trang "Trích Xuất Đặc Trưng"
2. Chọn model (MNIST hoặc Shape)
3. Upload ảnh
4. Chọn convolutional layer muốn visualize
5. Xem feature maps được trích xuất

## 🧠 Kiến Trúc CNN

### MNIST Model

```
Input: 28x28x1 (Grayscale)
    ↓
Conv2D(32, 3x3) + ReLU
    ↓
MaxPooling2D(2x2)
    ↓
BatchNormalization
    ↓
Conv2D(64, 3x3) + ReLU
    ↓
MaxPooling2D(2x2)
    ↓
BatchNormalization
    ↓
Conv2D(128, 3x3) + ReLU
    ↓
BatchNormalization
    ↓
Flatten
    ↓
Dropout(0.5)
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(10) + Softmax
    ↓
Output: 10 classes (0-9)
```

**Tổng Parameters**: ~200K  
**Training Accuracy**: >99%  
**Validation Accuracy**: >95%

### Shape Model

```
Input: 64x64x1 (Grayscale)
    ↓
2x [Conv2D(32, 3x3) + ReLU]
    ↓
MaxPooling2D(2x2) + BatchNorm + Dropout(0.25)
    ↓
2x [Conv2D(64, 3x3) + ReLU]
    ↓
MaxPooling2D(2x2) + BatchNorm + Dropout(0.25)
    ↓
2x [Conv2D(128, 3x3) + ReLU]
    ↓
MaxPooling2D(2x2) + BatchNorm + Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(8) + Softmax
    ↓
Output: 8 classes (Circle, Rectangle, Square, Triangle, Pentagon, Hexagon, Oval, Diamond)
```

**Tổng Parameters**: ~1.5M  
**Training Accuracy**: >98%  
**Validation Accuracy**: >95%

## 🖼️ Xử Lý Ảnh

### Pipeline Xử Lý

1. **Input**: Ảnh RGB/Grayscale từ user
2. **Grayscale Conversion**: Chuyển sang ảnh xám (nếu cần)
3. **Resize**: Resize về kích thước chuẩn
   - MNIST: 28x28
   - Shape: 64x64
4. **Normalization**: Chuẩn hóa pixel values về [0, 1]
5. **Reshape**: Thêm channel dimension (H, W, 1)
6. **Prediction**: Đưa vào CNN model

### Kỹ Thuật Xử Lý Ảnh

#### 1. Filters (Lọc ảnh)
- **Gaussian Blur**: Làm mịn ảnh, khử nhiễu
- **Median Filter**: Loại bỏ salt-and-pepper noise
- **Bilateral Filter**: Làm mịn nhưng giữ lại edges
- **Sharpening**: Tăng độ sắc nét

#### 2. Edge Detection (Phát hiện biên)
- **Canny**: Two-threshold edge detection
- **Sobel**: Gradient-based edge detection

#### 3. Segmentation (Phân đoạn)
- **Binary Threshold**: Ngưỡng cố định
- **Otsu Threshold**: Tự động tính ngưỡng tối ưu
- **Adaptive Threshold**: Ngưỡng thích ứng theo vùng
- **Watershed**: Phân đoạn dựa trên markers

## 📁 Cấu Trúc Dự Án

```
├── app.py                      # Main Streamlit application
├── image_processor.py          # Image processing utilities
│   ├── Filters (Gaussian, Median, Bilateral, Sharpen)
│   ├── Edge Detection (Canny, Sobel)
│   ├── Segmentation (Threshold, Otsu, Adaptive, Watershed)
│   └── Preprocessing functions
│
├── mnist_model.py              # MNIST CNN model
│   ├── Model architecture
│   ├── Training logic
│   ├── Prediction functions
│   └── Model persistence
│
├── shape_model.py              # Shape recognition CNN model
│   ├── Model architecture
│   ├── Training logic
│   ├── Prediction functions
│   └── Model persistence
│
├── shape_generator.py          # Synthetic shape dataset generator
│   ├── Shape creation functions (8 types)
│   ├── Data augmentation (rotation, noise)
│   └── Sample image generation
│
├── multi_object_detector.py    # Multi-object detection
│   ├── Contour detection
│   ├── Bounding box extraction
│   ├── Object classification
│   └── Visualization utilities
│
├── feature_visualizer.py       # Feature maps visualization
│   ├── Layer activation extraction
│   ├── Feature map plotting
│   ├── Training history plots
│   └── Confusion matrix
│
├── models/                     # Saved models directory
│   ├── mnist_cnn.keras
│   └── shape_cnn_64.keras
│
├── sample_images/              # Sample images for demo
│   ├── shapes/                 # Generated shape samples
│   └── mnist/                  # MNIST samples
│
└── README.md                   # This file
```

## 🔬 Chi Tiết Kỹ Thuật

### Dataset

#### MNIST
- **Source**: Keras datasets (built-in)
- **Size**: 70,000 ảnh (60K train, 10K test)
- **Classes**: 10 (digits 0-9)
- **Image size**: 28x28 grayscale
- **Format**: NumPy arrays

#### Shapes
- **Source**: Synthetically generated
- **Size**: Configurable (default 8,000 ảnh)
- **Classes**: 8 (Circle, Rectangle, Square, Triangle, Pentagon, Hexagon, Oval, Diamond)
- **Image size**: 64x64 grayscale
- **Augmentation**: Rotation, noise injection

### Training Configuration

#### MNIST
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10-15
- **Batch size**: 128
- **Callbacks**: Early Stopping, ReduceLROnPlateau

#### Shape Model
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30-50
- **Batch size**: 64
- **Callbacks**: Early Stopping, ReduceLROnPlateau
- **Data split**: 80% train, 20% validation

### Performance Optimization

1. **Batch Normalization**: Chuẩn hóa activations
2. **Dropout**: Prevent overfitting (0.25-0.5)
3. **Data Augmentation**: Rotation, noise cho shapes
4. **Early Stopping**: Dừng khi không cải thiện
5. **Learning Rate Scheduling**: Giảm LR khi plateau

## 📊 Kết Quả Thực Tế
**Pre-trained models hiện tại:**
### MNIST Model ✅
- **Validation Accuracy**: **99.2%** (vượt yêu cầu >95%)
- **Training epochs**: 5 epochs
- **Training time**: ~3 phút
- **Model size**: 2.9 MB
- **Dataset**: 60,000 training images, 10,000 test images
### Shape Model ✅
- **Validation Accuracy**: **92.5%** (gần đạt >95%)
- **Training epochs**: 12 epochs
- **Training time**: ~2 phút
- **Model size**: 2.4 MB
- **Dataset**: 4,800 synthetic training images (600/class), 960 test images
> **💡 Cải thiện Shape Model:** Retrain với epochs=30-50 và samples_per_class=1000-2000 để đạt >95% accuracy. Pre-trained model hiện tại đủ để demo và học tập.
**Khi retrain với cấu hình đầy đủ:**
- MNIST: 99%+ accuracy (10 epochs)
- Shape: 95-98% accuracy (30-50 epochs, 1000 samples/class)

## 🎯 Ứng Dụng Thực Tế

1. **Giáo dục**: Dạy học về CNN và Computer Vision
2. **OCR**: Optical Character Recognition cơ bản
3. **Automation**: Nhận dạng ký tự trong forms
4. **Geometry**: Phân tích hình học trong CAD
5. **Quality Control**: Kiểm tra hình dạng sản phẩm

## 🚀 Phát Triển Thêm

Các tính năng có thể mở rộng:

- [ ] Nhận dạng chữ cái (A-Z)
- [ ] Nhận dạng chữ viết tiếng Việt
- [ ] Object detection với YOLO/SSD
- [ ] Real-time detection qua webcam
- [ ] Transfer learning với pre-trained models
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] REST API cho integration
- [ ] Database lưu trữ kết quả
- [ ] User authentication
- [ ] Model versioning

## 📖 Tài Liệu Tham Khảo

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/api/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

## 📝 License

MIT License - Free to use for educational and commercial purposes.

## 👨‍💻 Thông Tin Dự Án

**Mục đích**: Dự án học tập về CNN và Computer Vision  
**Ngôn ngữ**: Python 3.11  
**Framework**: TensorFlow 2.x, Streamlit  
**Platform**: Replit

---

**Developed with ❤️ using TensorFlow & Streamlit**
