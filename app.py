import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import io
from streamlit_drawable_canvas import st_canvas

from image_processor import ImageProcessor
from mnist_model import MNISTModel
from shape_model import ShapeModel
from shape_generator import ShapeGenerator
from multi_object_detector import MultiObjectDetector
from feature_visualizer import FeatureVisualizer

st.set_page_config(
    page_title="Nhận Dạng Chữ Viết & Hình Học - CNN",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_mnist_model():
    model = MNISTModel()
    if not model.load_model():
        return None
    return model

@st.cache_resource
def load_shape_model():
    model = ShapeModel(input_size=64)
    if not model.load_model():
        return None
    return model

def check_models_trained():
    import os
    mnist_exists = os.path.exists('models/mnist_cnn.keras')
    shape_exists = os.path.exists('models/shape_cnn_64.keras')
    return mnist_exists, shape_exists

def main():
    st.markdown('<div class="main-header">🔍 Nhận Dạng Chữ Viết Tay & Hình Học</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sử dụng Mạng Neural Tích Chập (CNN) với Xử Lý Ảnh Nâng Cao</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("📋 Menu Điều Hướng")
        
        page = st.radio(
            "Chọn chức năng:",
            [
                "🏠 Trang Chủ",
                "🔢 Nhận Dạng Chữ Số (MNIST)",
                "🔷 Nhận Dạng Hình Học",
                "🎯 Phát Hiện Nhiều Đối Tượng",
                "🖼️ Xử Lý Ảnh Nâng Cao",
                "📊 Trích Xuất Đặc Trưng (Feature Maps)",
                "⚙️ Train Model",
                "📚 Hướng Dẫn Sử Dụng"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📖 Thông Tin Dự Án")
        st.markdown("""
        **Công Nghệ:**
        - TensorFlow/Keras
        - OpenCV
        - Streamlit
        - scikit-learn
        
        **Tính Năng:**
        - ✅ CNN nhận dạng MNIST
        - ✅ CNN nhận dạng 8+ hình
        - ✅ Phát hiện nhiều đối tượng
        - ✅ Xử lý ảnh nâng cao
        - ✅ Visualize feature maps
        - ✅ Export kết quả
        """)
    
    if page == "🏠 Trang Chủ":
        show_home_page()
    elif page == "🔢 Nhận Dạng Chữ Số (MNIST)":
        show_mnist_page()
    elif page == "🔷 Nhận Dạng Hình Học":
        show_shape_page()
    elif page == "🎯 Phát Hiện Nhiều Đối Tượng":
        show_multi_object_page()
    elif page == "🖼️ Xử Lý Ảnh Nâng Cao":
        show_image_processing_page()
    elif page == "📊 Trích Xuất Đặc Trưng (Feature Maps)":
        show_feature_maps_page()
    elif page == "⚙️ Train Model":
        show_training_page()
    elif page == "📚 Hướng Dẫn Sử Dụng":
        show_guide_page()

def show_home_page():
    mnist_trained, shape_trained = check_models_trained()
    
    col1, col2 = st.columns(2)
    with col1:
        if mnist_trained:
            st.metric("MNIST Model", "✅ Sẵn sàng", "99.2% accuracy")
        else:
            st.metric("MNIST Model", "❌ Chưa có", "Cần train")
    with col2:
        if shape_trained:
            st.metric("Shape Model", "✅ Sẵn sàng", "92.5% accuracy")
        else:
            st.metric("Shape Model", "❌ Chưa có", "Cần train")
    
    if mnist_trained and shape_trained:
        st.success("✅ Tất cả models đã sẵn sàng! Bắt đầu khám phá các tính năng nhận dạng ngay.")
        st.info("💡 **Mẹo:** Models đã được pre-train sẵn. Bạn có thể retrain để cải thiện accuracy hoặc học cách CNN hoạt động!")
    else:
        st.warning("⚠️ **Thiếu models!** Vui lòng vào trang **'⚙️ Train Model'** để train models trước.")
        st.info("""
        **Cách train nhanh:**
        1. Vào trang **'⚙️ Train Model'**
        2. Train MNIST (~2-3 phút, đạt 99% accuracy)
        3. Train Shape Model (~5-7 phút, đạt 95% accuracy)
        """)
        
        
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔢 MNIST")
        st.info("Nhận dạng chữ số viết tay từ 0-9 với độ chính xác >95%")
        st.markdown("**Tính năng:**")
        st.markdown("- Vẽ số trực tiếp")
        st.markdown("- Upload ảnh")
        st.markdown("- Batch processing")
        
    with col2:
        st.markdown("### 🔷 Hình Học")
        st.success("Nhận dạng 8+ hình: Tròn, Vuông, Chữ nhật, Tam giác, Ngũ giác, Lục giác, Oval, Hình thoi")
        st.markdown("**Tính năng:**")
        st.markdown("- Vẽ hình trực tiếp")
        st.markdown("- Upload ảnh")
        st.markdown("- Confidence score")
        
    with col3:
        st.markdown("### 🎯 Multi-Object")
        st.warning("Phát hiện nhiều đối tượng trong cùng một ảnh")
        st.markdown("**Tính năng:**")
        st.markdown("- Bounding boxes")
        st.markdown("- Đếm số lượng")
        st.markdown("- Export kết quả")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Quy Trình Xử Lý")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1️⃣ Input")
        st.markdown("📸 Upload/Vẽ ảnh")
    with col2:
        st.markdown("#### 2️⃣ Preprocessing")
        st.markdown("🔧 Resize, Normalize")
    with col3:
        st.markdown("#### 3️⃣ CNN")
        st.markdown("🧠 Trích xuất đặc trưng")
    with col4:
        st.markdown("#### 4️⃣ Output")
        st.markdown("✅ Kết quả nhận dạng")
    
    st.markdown("---")
    
    st.markdown("### 📊 Kiến Trúc CNN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### MNIST Model")
        st.code("""
Conv2D(32) -> MaxPool -> BatchNorm
Conv2D(64) -> MaxPool -> BatchNorm
Conv2D(128) -> BatchNorm
Flatten -> Dense(128) -> Dense(10)
        """)
        st.caption("Input: 28x28x1 | Output: 10 classes")
    
    with col2:
        st.markdown("#### Shape Model")
        st.code("""
Conv2D(32x2) -> MaxPool -> BatchNorm
Conv2D(64x2) -> MaxPool -> BatchNorm
Conv2D(128x2) -> MaxPool -> BatchNorm
Flatten -> Dense(256) -> Dense(8)
        """)
        st.caption("Input: 64x64x1 | Output: 8 classes")

def show_mnist_page():
    st.header("🔢 Nhận Dạng Chữ Số Viết Tay (MNIST)")
    
    model = load_mnist_model()
    if model is None:
        st.error("⚠️ **Model MNIST chưa được train!**")
        st.info("""
        **Để sử dụng tính năng này:**
        1. Vào trang **'⚙️ Train Model'** trong menu bên trái
        2. Click **'Bắt đầu Training MNIST'**
        3. Đợi ~2-3 phút để model train xong (độ chính xác >95%)
        4. Quay lại trang này để sử dụng!
        
        **Lưu ý:** Model chỉ cần train 1 lần duy nhất, sau đó sẽ được lưu lại.
        """)
        return
    
    tab1, tab2, tab3 = st.tabs(["✏️ Vẽ Tay", "📤 Upload Ảnh", "📦 Batch Processing"])
    
    with tab1:
        st.subheader("Vẽ chữ số từ 0-9")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=15,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="mnist_canvas",
            )
        
        with col2:
            if st.button("🔍 Nhận Dạng", key="predict_mnist_canvas", type="primary"):
                if canvas_result.image_data is not None:
                    input_image = canvas_result.image_data[:, :, 0]
                    
                    if np.sum(input_image) > 0:
                        preprocessed = ImageProcessor.preprocess_for_mnist(input_image)
                        
                        pred_class, confidence, probs = model.predict(preprocessed)
                        
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"### Kết Quả: **{pred_class}**")
                        st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("#### Phân Bố Xác Suất")
                        prob_df = pd.DataFrame({
                            'Số': list(range(10)),
                            'Xác suất (%)': probs * 100
                        })
                        st.bar_chart(prob_df.set_index('Số'))
                    else:
                        st.warning("Vui lòng vẽ một chữ số!")
    
    with tab2:
        st.subheader("Upload ảnh chứa chữ số")
        
        uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="mnist_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            preprocessed = ImageProcessor.preprocess_for_mnist(image)
            pred_class, confidence, probs = model.predict(preprocessed)
            
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Kết Quả: **{pred_class}**")
                st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")
                st.progress(float(confidence))
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### Top 3 Dự Đoán")
                top3_indices = np.argsort(probs)[-3:][::-1]
                for idx in top3_indices:
                    st.write(f"{idx}: {probs[idx]*100:.2f}%")
    
    with tab3:
        st.subheader("Xử lý nhiều ảnh cùng lúc")
        
        uploaded_files = st.file_uploader(
            "Upload nhiều ảnh", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True,
            key="mnist_batch"
        )
        
        if uploaded_files:
            if st.button("🔍 Nhận Dạng Tất Cả", type="primary"):
                results = []
                images_processed = []
                
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    image = Image.open(file)
                    preprocessed = ImageProcessor.preprocess_for_mnist(image)
                    images_processed.append(preprocessed)
                    
                    pred_class, confidence, _ = model.predict(preprocessed)
                    
                    results.append({
                        'Tên file': file.name,
                        'Kết quả': pred_class,
                        'Độ tin cậy (%)': f"{confidence*100:.2f}"
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"Đã xử lý {len(uploaded_files)} ảnh!")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải kết quả (CSV)",
                    data=csv,
                    file_name='mnist_results.csv',
                    mime='text/csv',
                )
                
                json_str = results_df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    label="📥 Tải kết quả (JSON)",
                    data=json_str,
                    file_name='mnist_results.json',
                    mime='application/json',
                )

def show_shape_page():
    st.header("🔷 Nhận Dạng Hình Học")
    
    model = load_shape_model()
    if model is None:
        st.error("⚠️ **Model Shape chưa được train!**")
        st.info("""
        **Để sử dụng tính năng này:**
        1. Vào trang **'⚙️ Train Model'** trong menu bên trái
        2. Click **'Bắt đầu Training Shape Model'**
        3. Đợi ~5-7 phút để model train xong (độ chính xác >95%)
        4. Quay lại trang này để sử dụng!
        
        **Lưu ý:** Model chỉ cần train 1 lần duy nhất, sau đó sẽ được lưu lại.
        """)
        return
    
    st.info("**8 loại hình:** Tròn, Chữ nhật, Vuông, Tam giác, Ngũ giác, Lục giác, Oval, Hình thoi")
    
    tab1, tab2, tab3 = st.tabs(["✏️ Vẽ Hình", "📤 Upload Ảnh", "🎨 Demo Ảnh Mẫu"])
    
    with tab1:
        st.subheader("Vẽ hình học")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=3,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=320,
                width=320,
                drawing_mode="freedraw",
                key="shape_canvas",
            )
        
        with col2:
            if st.button("🔍 Nhận Dạng Hình", key="predict_shape_canvas", type="primary"):
                if canvas_result.image_data is not None:
                    input_image = canvas_result.image_data[:, :, 0]
                    
                    if np.sum(input_image) > 0:
                        preprocessed = ImageProcessor.preprocess_for_shapes(input_image, (64, 64))
                        
                        pred_class, shape_name, confidence, probs = model.predict(preprocessed)
                        
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"### Kết Quả: **{shape_name}**")
                        st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")
                        st.progress(float(confidence))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("#### Phân Bố Xác Suất")
                        prob_df = pd.DataFrame({
                            'Hình': [ShapeGenerator.SHAPE_CLASSES[i] for i in range(len(probs))],
                            'Xác suất (%)': probs * 100
                        })
                        st.bar_chart(prob_df.set_index('Hình'))
                    else:
                        st.warning("Vui lòng vẽ một hình!")
    
    with tab2:
        st.subheader("Upload ảnh chứa hình học")
        
        uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="shape_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            preprocessed = ImageProcessor.preprocess_for_shapes(image, (64, 64))
            pred_class, shape_name, confidence, probs = model.predict(preprocessed)
            
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Kết Quả: **{shape_name}**")
                st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")
                st.progress(float(confidence))
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### Top 3 Dự Đoán")
                top3_indices = np.argsort(probs)[-3:][::-1]
                for idx in top3_indices:
                    st.write(f"{ShapeGenerator.SHAPE_CLASSES[idx]}: {probs[idx]*100:.2f}%")
    
    with tab3:
        st.subheader("Demo với ảnh mẫu")
        
        import os
        sample_dir = 'sample_images/shapes'
        
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
            
            if sample_files:
                selected_sample = st.selectbox("Chọn ảnh mẫu", sample_files)
                
                if selected_sample:
                    image_path = os.path.join(sample_dir, selected_sample)
                    image = Image.open(image_path)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption=selected_sample, use_container_width=True)
                    
                    preprocessed = ImageProcessor.preprocess_for_shapes(image, (64, 64))
                    pred_class, shape_name, confidence, probs = model.predict(preprocessed)
                    
                    with col2:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"### Kết Quả: **{shape_name}**")
                        st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")
                        st.progress(float(confidence))
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Chưa có ảnh mẫu. Tạo ảnh mẫu ở trang 'Train Model'")
        else:
            st.warning("Chưa có thư mục ảnh mẫu. Tạo ảnh mẫu ở trang 'Train Model'")

def show_multi_object_page():
    st.header("🎯 Phát Hiện Nhiều Đối Tượng")
    
    shape_model = load_shape_model()
    if shape_model is None:
        st.error("⚠️ **Model Shape chưa được train!**")
        st.info("""
        **Tính năng này cần Shape Model để hoạt động.**
        
        Vui lòng vào trang **'⚙️ Train Model'** để train Shape Model trước (~5-7 phút).
        """)
        return
    
    st.info("Phát hiện và nhận dạng nhiều hình học trong cùng một ảnh")
    
    uploaded_file = st.file_uploader("Upload ảnh chứa nhiều hình", type=['png', 'jpg', 'jpeg'], key="multi_object")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)
        
        with st.spinner("Đang phát hiện đối tượng..."):
            def preprocess_for_detector(roi):
                return ImageProcessor.preprocess_for_shapes(roi, (64, 64))
            
            detector = MultiObjectDetector(shape_model, preprocess_for_detector, min_area=100)
            detections = detector.detect_objects(image_np)
            
            result_image = detector.draw_detections(image_np, detections)
        
        with col2:
            st.subheader(f"Kết quả ({len(detections)} đối tượng)")
            st.image(result_image, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Thống kê")
            summary = detector.get_detection_summary(detections)
            st.metric("Tổng số đối tượng", summary['total_objects'])
            
            st.markdown("**Phân loại:**")
            for shape_name, count in summary['objects_by_class'].items():
                st.write(f"- {shape_name}: {count}")
        
        with col2:
            st.subheader("📋 Chi tiết")
            if detections:
                details = []
                for i, det in enumerate(detections):
                    details.append({
                        'STT': i + 1,
                        'Loại hình': det['name'],
                        'Độ tin cậy (%)': f"{det['confidence']*100:.2f}",
                        'Vị trí (x,y,w,h)': f"({det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]})"
                    })
                
                details_df = pd.DataFrame(details)
                st.dataframe(details_df, use_container_width=True)
                
                csv = details_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải kết quả (CSV)",
                    data=csv,
                    file_name='detection_results.csv',
                    mime='text/csv',
                )

def show_image_processing_page():
    st.header("🖼️ Xử Lý Ảnh Nâng Cao")
    
    st.markdown("""
    Trang này demo các kỹ thuật xử lý ảnh được sử dụng trong preprocessing:
    - **Filters:** Làm mịn, khử nhiễu
    - **Edge Detection:** Phát hiện biên cạnh
    - **Segmentation:** Phân đoạn ảnh
    """)
    
    uploaded_file = st.file_uploader("Upload ảnh để xử lý", type=['png', 'jpg', 'jpeg'], key="image_processing")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["🔧 Filters", "📐 Edge Detection", "🎨 Segmentation"])
        
        with tab1:
            st.subheader("Bộ lọc ảnh")
            
            filter_type = st.selectbox(
                "Chọn bộ lọc",
                ["Gaussian Blur", "Median Filter", "Bilateral Filter", "Sharpen"]
            )
            
            gray = ImageProcessor.convert_to_grayscale(image_np)
            
            if filter_type == "Gaussian Blur":
                kernel_size = st.slider("Kernel size", 3, 15, 5, step=2)
                processed = ImageProcessor.apply_gaussian_blur(gray, kernel_size)
            elif filter_type == "Median Filter":
                kernel_size = st.slider("Kernel size", 3, 15, 5, step=2)
                processed = ImageProcessor.apply_median_filter(gray, kernel_size)
            elif filter_type == "Bilateral Filter":
                processed = ImageProcessor.apply_bilateral_filter(gray)
            else:
                processed = ImageProcessor.sharpen_image(gray)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Ảnh gốc (Grayscale)", use_container_width=True)
            with col2:
                st.image(processed, caption=f"Sau {filter_type}", use_container_width=True)
        
        with tab2:
            st.subheader("Phát hiện biên")
            
            edge_type = st.selectbox(
                "Chọn phương pháp",
                ["Canny", "Sobel"]
            )
            
            gray = ImageProcessor.convert_to_grayscale(image_np)
            
            if edge_type == "Canny":
                col1, col2 = st.columns(2)
                with col1:
                    threshold1 = st.slider("Threshold 1", 0, 255, 100)
                with col2:
                    threshold2 = st.slider("Threshold 2", 0, 255, 200)
                edges = ImageProcessor.detect_edges_canny(gray, threshold1, threshold2)
            else:
                edges = ImageProcessor.detect_edges_sobel(gray)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(edges, caption=f"Edges ({edge_type})", use_container_width=True)
        
        with tab3:
            st.subheader("Phân đoạn ảnh")
            
            seg_type = st.selectbox(
                "Chọn phương pháp",
                ["Binary Threshold", "Otsu Threshold", "Adaptive Threshold", "Watershed"]
            )
            
            gray = ImageProcessor.convert_to_grayscale(image_np)
            
            if seg_type == "Binary Threshold":
                threshold_val = st.slider("Threshold", 0, 255, 127)
                segmented = ImageProcessor.threshold_binary(gray, threshold_val)
            elif seg_type == "Otsu Threshold":
                segmented = ImageProcessor.threshold_otsu(gray)
            elif seg_type == "Adaptive Threshold":
                block_size = st.slider("Block size", 3, 21, 11, step=2)
                segmented = ImageProcessor.adaptive_threshold(gray, block_size)
            else:
                segmented = ImageProcessor.watershed_segmentation(image_np)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(segmented, caption=f"Segmented ({seg_type})", use_container_width=True)

def show_feature_maps_page():
    st.header("📊 Trích Xuất Đặc Trưng (Feature Maps)")
    
    st.info("Visualize cách CNN học và trích xuất đặc trưng từ ảnh")
    
    model_type = st.radio("Chọn model", ["MNIST", "Shape"])
    
    if model_type == "MNIST":
        model = load_mnist_model()
        preprocess_func = ImageProcessor.preprocess_for_mnist
    else:
        model = load_shape_model()
        preprocess_func = lambda img: ImageProcessor.preprocess_for_shapes(img, (64, 64))
    
    if model is None:
        st.error("Model chưa sẵn sàng.")
        st.error(f"⚠️ **Model {model_type} chưa được train!**")
        st.info("""
        **Để sử dụng tính năng này:**
        Vui lòng vào trang **'⚙️ Train Model'** để train model trước.
        """)
        return
    
    uploaded_file = st.file_uploader("Upload ảnh", type=['png', 'jpg', 'jpeg'], key="feature_maps")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh đầu vào")
            st.image(image, use_container_width=True)
        
        preprocessed = preprocess_func(image)
        
        with col2:
            st.subheader("Ảnh sau preprocessing")
            st.image(preprocessed.squeeze(), use_container_width=True, clamp=True)
        
        st.markdown("---")
        
        visualizer = FeatureVisualizer(model.model)
        conv_layers = visualizer.get_conv_layers()
        
        if conv_layers:
            selected_layer = st.selectbox("Chọn layer để visualize", conv_layers)
            
            layer_index = conv_layers.index(selected_layer)
            
            max_filters = st.slider("Số lượng filters hiển thị", 4, 32, 16, step=4)
            
            with st.spinner("Đang tạo feature maps..."):
                fig = visualizer.visualize_feature_maps(preprocessed, layer_index, max_filters)
                st.pyplot(fig)
            
            if st.checkbox("Hiển thị tất cả layers"):
                with st.spinner("Đang tạo visualization cho tất cả layers..."):
                    figures = visualizer.visualize_all_layers(preprocessed, max_filters_per_layer=8)
                    
                    for fig in figures:
                        st.pyplot(fig)
        else:
            st.warning("Không tìm thấy convolutional layers trong model")

def show_training_page():
    st.header("⚙️ Train Model")
    
    st.warning("⏰ Training có thể mất vài phút. Vui lòng đợi cho đến khi hoàn tất.")
    
    tab1, tab2, tab3 = st.tabs(["🔢 Train MNIST", "🔷 Train Shape Model", "🎨 Tạo Ảnh Mẫu"])
    
    with tab1:
        st.subheader("Train MNIST Model")
        
        mnist_trained, _ = check_models_trained()
        if mnist_trained:
            st.success("✅ Model MNIST đã được train! Bạn có thể train lại để cải thiện hoặc thử tham số khác.")
        else:
            st.info("⚠️ Model chưa được train. Hãy train ngay để sử dụng tính năng nhận dạng chữ số!")
        
        st.markdown("""
        **Dataset:** MNIST (70,000 ảnh chữ số viết tay)
        - Training: 60,000 ảnh
        - Test: 10,000 ảnh
        - Classes: 0-9 (10 classes)
        - **Thời gian:** ~2-3 phút
        - **Độ chính xác mong đợi:** >95%
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            epochs_mnist = st.number_input("Số epochs", 1, 50, 10, key="mnist_epochs")
        with col2:
            batch_size_mnist = st.number_input("Batch size", 32, 256, 128, key="mnist_batch")
        
        if st.button("🚀 Bắt đầu Training MNIST", type="primary"):
            with st.spinner("Đang training model..."):
                model = MNISTModel()
                
                progress_text = st.empty()
                progress_text.text("Đang load dataset...")
                
                history = model.train(epochs=epochs_mnist, batch_size=batch_size_mnist)
                
                progress_text.text("Training hoàn tất!")
                
                st.success("✅ Training MNIST model thành công!")
                
                visualizer = FeatureVisualizer(model.model)
                fig = visualizer.plot_training_history(history)
                st.pyplot(fig)
                
                final_acc = history.history['val_accuracy'][-1]
                st.metric("Validation Accuracy", f"{final_acc*100:.2f}%")
    
    with tab2:
        st.subheader("Train Shape Recognition Model")
        
        _, shape_trained = check_models_trained()
        if shape_trained:
            st.success("✅ Model Shape đã được train! Bạn có thể train lại để cải thiện hoặc thử tham số khác.")
        else:
            st.info("⚠️ Model chưa được train. Hãy train ngay để sử dụng tính năng nhận dạng hình học!")
        
        st.markdown("""
        **Dataset:** Synthetic Shapes (tự tạo)
        - Shapes: Circle, Rectangle, Square, Triangle, Pentagon, Hexagon, Oval, Diamond
        - Augmentation: Rotation, Noise
        - **Thời gian:** ~5-7 phút
        - **Độ chính xác mong đợi:** >95%
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs_shape = st.number_input("Số epochs", 1, 50, 30, key="shape_epochs")
        with col2:
            batch_size_shape = st.number_input("Batch size", 32, 128, 64, key="shape_batch")
        with col3:
            samples_per_class = st.number_input("Samples/class", 500, 2000, 1000, key="samples")
        
        if st.button("🚀 Bắt đầu Training Shape Model", type="primary"):
            with st.spinner("Đang training model..."):
                model = ShapeModel(input_size=64)
                
                progress_text = st.empty()
                progress_text.text("Đang tạo dataset...")
                
                history = model.train(
                    epochs=epochs_shape,
                    batch_size=batch_size_shape,
                    samples_per_class=samples_per_class
                )
                
                progress_text.text("Training hoàn tất!")
                
                st.success("✅ Training Shape model thành công!")
                
                visualizer = FeatureVisualizer(model.model)
                fig = visualizer.plot_training_history(history)
                st.pyplot(fig)
                
                final_acc = history.history['val_accuracy'][-1]
                st.metric("Validation Accuracy", f"{final_acc*100:.2f}%")
    
    with tab3:
        st.subheader("Tạo Ảnh Mẫu")
        
        st.markdown("Tạo ảnh mẫu cho demo và testing")
        
        num_samples = st.number_input("Số ảnh mẫu mỗi loại hình", 1, 10, 5)
        
        if st.button("🎨 Tạo Ảnh Mẫu", type="primary"):
            with st.spinner("Đang tạo ảnh mẫu..."):
                ShapeGenerator.save_sample_images(samples=num_samples)
                st.success(f"✅ Đã tạo {num_samples * 8} ảnh mẫu tại sample_images/shapes/")
                
                import os
                sample_dir = 'sample_images/shapes'
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')][:8]
                
                cols = st.columns(4)
                for i, file in enumerate(sample_files):
                    with cols[i % 4]:
                        img = Image.open(os.path.join(sample_dir, file))
                        st.image(img, caption=file, use_container_width=True)

def show_guide_page():
    st.header("📚 Hướng Dẫn Sử Dụng")
    
    st.markdown("""
    ## 🎯 Quy Trình Sử Dụng
    
    ### Bước 1: Train Models
    1. Vào trang **"⚙️ Train Model"**
    2. Train MNIST model (10 epochs, ~2-3 phút)
    3. Train Shape model (30 epochs, ~5-7 phút)
    4. Tạo ảnh mẫu cho demo
    
    ### Bước 2: Sử Dụng Tính Năng
    
    #### 🔢 Nhận Dạng Chữ Số (MNIST)
    - **Vẽ tay:** Vẽ số từ 0-9 trên canvas
    - **Upload:** Upload ảnh chứa chữ số
    - **Batch:** Xử lý nhiều ảnh cùng lúc
    
    #### 🔷 Nhận Dạng Hình Học
    - Nhận dạng 8 loại hình: Tròn, Vuông, Chữ nhật, Tam giác, Ngũ giác, Lục giác, Oval, Hình thoi
    - Vẽ hoặc upload ảnh
    - Xem demo với ảnh mẫu
    
    #### 🎯 Phát Hiện Nhiều Đối Tượng
    - Upload ảnh chứa nhiều hình
    - Tự động phát hiện và vẽ bounding boxes
    - Export kết quả dưới dạng CSV
    
    #### 🖼️ Xử Lý Ảnh Nâng Cao
    - **Filters:** Gaussian, Median, Bilateral, Sharpen
    - **Edge Detection:** Canny, Sobel
    - **Segmentation:** Binary, Otsu, Adaptive, Watershed
    
    #### 📊 Feature Maps
    - Visualize cách CNN học đặc trưng
    - Xem feature maps từ các convolutional layers
    - Hiểu quá trình trích xuất đặc trưng
    
    ---
    
    ## 🔬 Kiến Trúc CNN
    
    ### MNIST Model
    ```
    Input (28x28x1)
    ↓
    Conv2D(32) + ReLU → MaxPool → BatchNorm
    ↓
    Conv2D(64) + ReLU → MaxPool → BatchNorm
    ↓
    Conv2D(128) + ReLU → BatchNorm
    ↓
    Flatten → Dense(128) → Dropout → Dense(10) + Softmax
    ```
    
    ### Shape Model
    ```
    Input (64x64x1)
    ↓
    2x Conv2D(32) + ReLU → MaxPool → BatchNorm → Dropout
    ↓
    2x Conv2D(64) + ReLU → MaxPool → BatchNorm → Dropout
    ↓
    2x Conv2D(128) + ReLU → MaxPool → BatchNorm → Dropout
    ↓
    Flatten → Dense(256) → Dropout → Dense(8) + Softmax
    ```
    
    ---
    
    ## 📊 Xử Lý Ảnh Pipeline
    
    1. **Input:** Ảnh gốc (RGB/Grayscale)
    2. **Grayscale:** Chuyển sang ảnh xám
    3. **Resize:** Resize về kích thước chuẩn (28x28 hoặc 64x64)
    4. **Normalize:** Chuẩn hóa pixel values về [0, 1]
    5. **Reshape:** Thêm channel dimension
    6. **Predict:** Đưa vào CNN model
    
    ---
    
    ## 💡 Tips & Tricks
    
    ### MNIST
    - Vẽ số to, rõ ràng
    - Tránh vẽ quá nhiều nét
    - Số nên nằm ở giữa canvas
    
    ### Shape Detection
    - Vẽ hình đơn giản, rõ ràng
    - Tránh vẽ các hình chồng lên nhau
    - Hình nên có kích thước vừa phải
    
    ### Multi-Object Detection
    - Upload ảnh có background đơn giản
    - Các hình nên cách nhau rõ ràng
    - Tránh các hình quá nhỏ
    
    ---
    
    ## 🛠️ Cấu Trúc Code
    
    ```
    ├── app.py                      # Main Streamlit app
    ├── image_processor.py          # Xử lý ảnh (filters, edges, segmentation)
    ├── mnist_model.py              # MNIST CNN model
    ├── shape_model.py              # Shape recognition CNN model
    ├── shape_generator.py          # Tạo synthetic shape dataset
    ├── multi_object_detector.py    # Phát hiện nhiều đối tượng
    ├── feature_visualizer.py       # Visualize feature maps
    ├── models/                     # Saved models
    └── sample_images/              # Ảnh mẫu demo
    ```
    
    ---
    
    ## 📖 Tài Liệu Tham Khảo
    
    - **TensorFlow/Keras:** https://www.tensorflow.org/
    - **OpenCV:** https://opencv.org/
    - **Streamlit:** https://streamlit.io/
    - **MNIST Dataset:** http://yann.lecun.com/exdb/mnist/
    
    ---
    
    ## 👨‍💻 Phát Triển Thêm
    
    Có thể mở rộng dự án với:
    - ✨ Nhận dạng chữ cái (A-Z)
    - ✨ Nhận dạng chữ viết tiếng Việt
    - ✨ Object detection với YOLO/SSD
    - ✨ Data augmentation nâng cao
    - ✨ Transfer learning với pre-trained models
    - ✨ Real-time detection qua webcam
    """)

if __name__ == "__main__":
    main()
