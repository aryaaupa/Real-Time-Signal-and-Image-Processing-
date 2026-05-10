import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft

st.set_page_config(page_title="Real-Time Signal & Image Processing", layout="wide")

st.title("Real-Time Signal and Image Processing Toolkit")
st.write("Interactive toolkit for image enhancement, edge detection, denoising, and signal frequency analysis.")

tab1, tab2 = st.tabs(["Image Processing", "Signal Processing"])

with tab1:
    st.header("Image Processing")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        option = st.selectbox(
            "Choose processing technique",
            ["Grayscale", "Canny Edge Detection", "Gaussian Blur", "Sobel Filter"]
        )

        if option == "Grayscale":
            processed = gray

        elif option == "Canny Edge Detection":
            processed = cv2.Canny(gray, 100, 200)

        elif option == "Gaussian Blur":
            processed = cv2.GaussianBlur(img_array, (15, 15), 0)

        elif option == "Sobel Filter":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            processed = cv2.magnitude(sobelx, sobely)

        with col2:
            st.subheader("Processed Output")
            st.image(processed, use_container_width=True, clamp=True)

with tab2:
    st.header("Signal Processing")

    st.write("Generate a sample signal and analyze its frequency spectrum.")

    frequency = st.slider("Signal frequency", 1, 50, 10)
    sampling_rate = 500
    duration = 1

    t = np.linspace(0, duration, sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t)

    fft_values = np.abs(fft(signal))

    fig1, ax1 = plt.subplots()
    ax1.plot(t, signal)
    ax1.set_title("Time-Domain Signal")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(fft_values[:sampling_rate // 2])
    ax2.set_title("Frequency Spectrum")
    ax2.set_xlabel("Frequency Bin")
    ax2.set_ylabel("Magnitude")
    st.pyplot(fig2)
