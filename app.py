import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Image Manager", layout="centered")
st.title("📸 Image Manager")

# Helper function to convert image to downloadable format
def convert_to_downloadable(img, format="png"):
    img_pil = Image.fromarray(img)
    buf = BytesIO()
    img_pil.save(buf, format=format.upper())
    byte_im = buf.getvalue()
    return byte_im

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(display_image, channels="RGB")

    operation = st.selectbox("Select an operation", [
        "None",
        "Convert to Grayscale",
        "Resize",
        "Blur",
        "Rotate",
        "Crop",
        "Analyze Contrast",
        "Analyze Exposure",
        "Sharpness Analysis",
        "Download Processed Image"
    ])

    processed_image = image.copy()

    if operation == "Convert to Grayscale":
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(processed_image, caption="Grayscale Image", channels="GRAY")

        # Convert to RGB for consistent saving
        downloadable_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        st.download_button(
        label="Download Grayscale Image",
        data=convert_to_downloadable(downloadable_image),
        file_name="grayscale_image.png",
        mime="image/png")

    elif operation == "Resize":
        width = st.number_input("Width", value=image.shape[1])
        height = st.number_input("Height", value=image.shape[0])
        if st.button("Resize"):
            processed_image = cv2.resize(image, (int(width), int(height)))
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Resized Image", channels="RGB")
            
            st.download_button(
            label="Download Resized Image",
            data=convert_to_downloadable(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)),
            file_name="resized_image.png",
            mime="image/png")

    elif operation == "Blur":
        ksize = st.slider("Kernel Size (odd only)", min_value=1, max_value=21, step=2, value=5)
        if st.button("Blur"):
            processed_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Blurred Image", channels="RGB")
            
            st.download_button(
            label="Download Blurred Image",
            data=convert_to_downloadable(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)),
            file_name="blurred_image.png",
            mime="image/png")

    elif operation == "Rotate":
        angle = st.slider("Rotation Angle", -180, 180, 0)
        if st.button("Rotate"):
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed_image = cv2.warpAffine(image, M, (w, h))
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Rotated Image", channels="RGB")
            
            st.download_button(
            label="Download Rotated Image",
            data=convert_to_downloadable(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)),
            file_name="rotated_image.png",
            mime="image/png")

    elif operation == "Crop":
        h, w = image.shape[:2]
        x1 = st.slider("Start X", 0, w, 0)
        y1 = st.slider("Start Y", 0, h, 0)
        x2 = st.slider("End X", 0, w, w)
        y2 = st.slider("End Y", 0, h, h)
        if st.button("Crop"):
            processed_image = image[y1:y2, x1:x2]
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Cropped Image", channels="RGB")
            st.download_button(
            label="Download Cropped Image",
            data=convert_to_downloadable(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)),
            file_name="cropped_image.png",
            mime="image/png")

    elif operation == "Analyze Contrast":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        st.metric("Contrast (Standard Deviation)", f"{contrast:.2f}")

    elif operation == "Analyze Exposure":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exposure = gray.mean()
        st.metric("Exposure (Mean Brightness)", f"{exposure:.2f}")

    elif operation == "Sharpness Analysis":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        st.metric("Sharpness (Laplacian Variance)", f"{laplacian_var:.2f}")

    elif operation == "Download Processed Image":
        if processed_image.ndim == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        download_button = st.download_button(
            label="Download Image",
            data=convert_to_downloadable(processed_image),
            file_name="processed_image.png",
            mime="image/png"
        )
