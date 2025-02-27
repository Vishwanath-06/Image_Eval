import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from utils import preprocess_image, resize_images_to_same_size, cosine_similarity, orb_feature_compare, compare_histograms, compare_ssim, mse

# Streamlit UI
st.title("🔍 Image Comparison App")
st.write("Upload two images to compare their similarity.")

uploaded_file1 = st.file_uploader("Upload first image", type=["jpg", "png", "webp"])
uploaded_file2 = st.file_uploader("Upload second image", type=["jpg", "png", "webp"])

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    # Convert images to OpenCV format
    imageA_cv = np.array(image1)
    imageB_cv = np.array(image2)

    # Ensure correct format for OpenCV processing
    if imageA_cv.shape[-1] == 4:
        imageA_cv = cv2.cvtColor(imageA_cv, cv2.COLOR_RGBA2RGB)
    if imageB_cv.shape[-1] == 4:
        imageB_cv = cv2.cvtColor(imageB_cv, cv2.COLOR_RGBA2RGB)

    # Resize images
    imageA_cv, imageB_cv = resize_images_to_same_size(imageA_cv, imageB_cv)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="First Image", use_column_width=True)
    with col2:
        st.image(image2, caption="Second Image", use_column_width=True)

    # Compute similarity scores
    cos_sim = cosine_similarity(preprocess_image(image1), preprocess_image(image2))
    orb_matches, _ = orb_feature_compare(imageA_cv, imageB_cv)
    hist_score = compare_histograms(imageA_cv, imageB_cv, method='correlation')
    ssim_score = compare_ssim(cv2.cvtColor(imageA_cv, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB_cv, cv2.COLOR_BGR2GRAY))
    mse_score = mse(cv2.cvtColor(imageA_cv, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB_cv, cv2.COLOR_BGR2GRAY))

    # Display results
    st.subheader("🔹 Similarity Scores")
    st.write(f"**Cosine Similarity (ResNet):** {cos_sim:.4f}")
    st.write(f"**ORB Feature Matches:** {orb_matches}")
    st.write(f"**Histogram Comparison (Correlation):** {hist_score:.4f}")
    st.write(f"**SSIM Score:** {ssim_score:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse_score:.4f}")
