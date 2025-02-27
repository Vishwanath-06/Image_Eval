import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])

# Image Preprocessing
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# Resize images without distortion
def resize_images_to_same_size(imageA, imageB):
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]

    new_h = max(h1, h2)
    new_w = max(w1, w2)

    def add_padding(img, target_h, target_w):
        h, w = img.shape[:2]
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return add_padding(imageA, new_h, new_w), add_padding(imageB, new_h, new_w)

# Cosine Similarity
def cosine_similarity(featureA, featureB):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(featureA, featureB).item()

# ORB Feature Matching
def orb_feature_compare(imageA, imageB):
    orb = cv2.ORB_create()
    kpA, desA = orb.detectAndCompute(imageA, None)
    kpB, desB = orb.detectAndCompute(imageB, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    return len(matches), sorted(matches, key=lambda x: x.distance)

# Histogram Comparison
def compare_histograms(imageA, imageB, method='correlation'):
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    cv2.normalize(histA, histA, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

# SSIM
def compare_ssim(imageA, imageB):
    return ssim(imageA, imageB, full=True)[0]

# Mean Squared Error (MSE)
def mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
