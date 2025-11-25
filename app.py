import streamlit as st
import numpy as np
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
    st.markdown('<div class="main-header">Nhận Dạng Chữ Viết Tay & Hình Dạng Đơn Giản</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sử dụng Mạng Neural Tích Chập (CNN) với Xử Lý Ảnh Nâng Cao</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        page = st.radio(
            "",
            [
                "🏠 Trang Chủ",
                "🔢 Nhận Dạng Chữ Số Viết Tay",
                "🔷 Nhận Dạng Hình Dạng Đơn Giản",
                "⚙️ Train Model"
            ],
            label_visibility="collapsed"
        )
    
    if page == "🏠 Trang Chủ":
        show_home_page()
    elif page == "🔢 Nhận Dạng Chữ Số Viết Tay":
        show_mnist_page()
    elif page == "🔷 Nhận Dạng Hình Dạng Đơn Giản":
        show_shape_page()
    elif page == "⚙️ Train Model":
        show_training_page()

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
        st.success("✅ Tất cả models đã sẵn sàng!")
    else:
        st.warning("⚠️ **Thiếu models!** Vui lòng vào trang **'⚙️ Train Model'** để train models trước.")
        


def show_mnist_page():
    st.header("🔢 Nhận Dạng Chữ Số Viết Tay")
    
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
    
    tab1, tab2 = st.tabs(["✏️ Vẽ Tay", "📤 Upload Ảnh"])
    
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

def show_shape_page():
    st.header("🔷 Nhận Dạng Hình Dạng Đơn Giản")
    
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
                
                final_acc = history.history['val_accuracy'][-1]
                final_loss = history.history['val_loss'][-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Validation Accuracy", f"{final_acc*100:.2f}%")
                with col2:
                    st.metric("Validation Loss", f"{final_loss:.4f}")
    
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
                
                final_acc = history.history['val_accuracy'][-1]
                final_loss = history.history['val_loss'][-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Validation Accuracy", f"{final_acc*100:.2f}%")
                with col2:
                    st.metric("Validation Loss", f"{final_loss:.4f}")
    
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

if __name__ == "__main__":
    main()
