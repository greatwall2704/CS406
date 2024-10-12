from PIL import Image
import numpy as np
import cv2
import streamlit as st
def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img.convert('RGB'))
    return img

def add_gaussian_noise(image):
    # Thêm nhiễu Gaussian
    mean = 0
    std = 25  # Độ lệch chuẩn của nhiễu
    gaussian_noise = np.random.normal(mean, std, image.shape)

    # Thêm nhiễu vào ảnh (cộng từng pixel với giá trị nhiễu tương ứng)
    noisy_image = image + gaussian_noise

    # Clip để đảm bảo giá trị pixel nằm trong khoảng [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def denoise(image):
    # Denoise with Mean
    mean_denoised = cv2.blur(image, (5, 5))
    
    #Denoise with Median
    median_denoised = cv2.medianBlur(image, 5)
    
    #Denoise Bilateral filter
    bilateral_denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    return mean_denoised, median_denoised, bilateral_denoised

def sharpen(image):
    '''
    1 vài kernel làm sharpen ảnh
    S1 = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]])

    S2 = np.array([[-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]])

    S3 = np.array([[1, -2, 1],
                [-2, 5, -2],
                [1, -2, 1]])
    '''
    kernel_1 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    
    kernel_2 = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])
    # Use cv2.filter2D to apply the kernel to the image
    sharpened_1 = cv2.filter2D(image, -1, kernel_1)
    sharpened_2 = cv2.filter2D(image, -1, kernel_2)
    
    # Sharpen with USM( Unsharp Masking)
    blurred_image = cv2.GaussianBlur(image, (5,5), 1)
    sharpened_3 = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    sharpened_4 = cv2.addWeighted(image, 2.5, blurred_image, -1.5, 0)

    return sharpened_1, sharpened_2, sharpened_3, sharpened_4

def edge_detection(image):
    # Sobel Edge Detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel_combined)
    
    #Prewitt Edge Detection
    kernel_prewitt_x = np.array([[ -1, 0, 1], 
                                [ -1, 0, 1], 
                                [ -1, 0, 1]])

    kernel_prewitt_y = np.array([[ -1, -1, -1], 
                                [  0,  0,  0], 
                                [  1,  1,  1]])
    prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
    prewitt = prewitt_x + prewitt_y
    
    # Canny Edge Detection
    canny = cv2.Canny(image, threshold1=100, threshold2=200)

    return sobel, prewitt, canny

st.set_page_config(layout="wide")

st.write("<div style='text-align: center; font-size:48px; font-weight: bold; padding-bottom: 16px'>Image Enhancing</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_image(uploaded_file)
    
    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Denoising:</div>", unsafe_allow_html=True)    
    blurred_image = cv2.GaussianBlur(img, (15, 15), 0)

    #Noisy with Gaussian
    noisy_img = add_gaussian_noise(img)
        
    cols_denoised = st.columns(5)
    
    mean_denoised, median_denoised, bilateral_denoised = denoise(img)

    cols_denoised[0].image(img, caption="Original")
        
    cols_denoised[1].image(noisy_img, caption="Noisy(Gaussian)")
        
    cols_denoised[2].image(mean_denoised, caption="Denoised(Mean)")
        
    cols_denoised[3].image(median_denoised, caption="Denoised(Median)")
    
    cols_denoised[4].image(bilateral_denoised, caption='Denoised(Bilateral)')
    
        
    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Sharpening:</div>", unsafe_allow_html=True)    

    cols_sharpened = st.columns(5)

    sharpened_1, sharpened_2, sharpened_3, sharpened_4 = sharpen(img)

    cols_sharpened[0].image(img, caption='Original')
    
    cols_sharpened[1].image(sharpened_1, caption='Sharpened(Kernel)')
    
    cols_sharpened[2].image(sharpened_2, caption='Sharpened(Kernel)')

    cols_sharpened[3].image(sharpened_3, caption='Sharpened(USM)')

    cols_sharpened[4].image(sharpened_4, caption='Sharpened(USM')

    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Edge Detection Filter:</div>", unsafe_allow_html=True)    

    cols_edge = st.columns(4)
    
    sobel, prewitt, canny_edges = edge_detection(img)
    
    cols_edge[0].image(sharpened_3, caption='Original')
    
    cols_edge[1].image(sobel, caption='Sobel')
    
    cols_edge[2].image(prewitt, caption='Prewitt')
    
    cols_edge[3].image(canny_edges, caption='Canny')