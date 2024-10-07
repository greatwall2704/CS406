import cv2
import os
import random
import numpy as np
from glob import glob
import streamlit as st
import pickle

# Biến toàn cục để lưu histogram của các ảnh trong tập seg
seg_hist_list_cached = None

color_space_dict = {
    "buildings": "bgr",
    "mountain": "bgr",
    "street": "bgr",
    "forest": "hsv",
    "glacier": "hsv",
    "sea": "hsv"
}

# Hàm tính histogram của ảnh
def calculate_histogram(image, image_type):
    color_space = color_space_dict[image_type]
    hist = []
    if color_space == "bgr":
        # Tính histogram cho từng kênh màu
        hist.append(cv2.calcHist([image], [0], None, [256], [0, 256]))
        hist.append(cv2.calcHist([image], [1], None, [256], [0, 256]))
        hist.append(cv2.calcHist([image], [2], None, [256], [0, 256]))
        
    elif color_space == "hsv":
        # Chuyển đổi ảnh sang không gian màu hsv
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Tính histogram cho ảnh hsv
        hist.append(cv2.calcHist([hsv_image], [0], None, [256], [0, 256]))
        hist.append(cv2.calcHist([hsv_image], [1], None, [256], [0, 256]))
        hist.append(cv2.calcHist([hsv_image], [2], None, [256], [0, 256]))
        
        
    hist = np.concatenate(hist)
    return cv2.normalize(hist, hist).flatten()

# Hàm so khớp histogram
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def pick_random_image(folder_path):
  # Danh sách các folder con
  subfolders = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
  # subfolders = ['street']

  # Chọn ngẫu nhiên một folder con
  selected_folder = random.choice(subfolders)

  folder_choice_path = os.path.join(folder_path, selected_folder)
  # Lấy danh sách các ảnh trong folder con đã chọn
  images = os.listdir(folder_choice_path)

  # Chọn ngẫu nhiên một ảnh từ danh sách
  selected_image = random.choice(images)

  # Kết quả
  return os.path.join(folder_choice_path, selected_image)

# Hàm lấy tên ảnh
def take_folder_name(img_path):
    return os.path.basename(os.path.dirname(img_path))

def calc_hist_seg_image(seg_path, cache_file='seg_hist_list.pkl'):
    global seg_hist_list_cached  # Sử dụng biến toàn cục để lưu cache
    
    # Kiểm tra xem file cache đã tồn tại chưa
    if os.path.exists(cache_file):
        # Đọc từ file cache
        with open(cache_file, 'rb') as f:
            seg_hist_list_cached = pickle.load(f)
    else:  # Chỉ tính toán lại nếu chưa có cache
        seg_hist_list_cached = []
        
        # Duyệt qua tất cả các ảnh trong thư mục seg
        image_paths = glob(os.path.join(seg_path, '**', '*.jpg'), recursive=True)

        for image_path in image_paths:
            # Đọc ảnh từ tập seg
            seg_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            seg_image = cv2.resize(seg_image, (256, 256))

            # Tính histogram của ảnh
            seg_hist = calculate_histogram(seg_image, take_folder_name(image_path))
            
            # Lưu histogram và đường dẫn của ảnh vào mảng
            seg_hist_list_cached.append((seg_hist, image_path))
        
        # Lưu kết quả vào file cache
        with open(cache_file, 'wb') as f:
            pickle.dump(seg_hist_list_cached, f)
    
    return seg_hist_list_cached

# Hàm tìm 10 ảnh giống nhất
def find_similar_images(input_image_path, seg_hist_list):
    # Đọc ảnh đầu vào
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    input_image = cv2.resize(input_image, (256, 256))
    
    # Tính histogram của ảnh đầu vào
    input_hist = calculate_histogram(input_image, take_folder_name(input_image_path))
    
    # Tạo danh sách để lưu khoảng cách và đường dẫn ảnh
    similarity_list = []
        
    # Tính độ tương đồng giữa histogram của ảnh đầu vào và các ảnh trong tập seg
    for seg_hist, seg_path in seg_hist_list:
        similarity = compare_histograms(input_hist, seg_hist)
        
        # Lưu độ tương đồng giữa histogram và đường dẫn ảnh với mảng
        similarity_list.append((similarity, seg_path))

    # Sắp xếp danh sách theo độ tương đồng giảm dần
    similarity_list.sort(reverse=True, key=lambda x: x[0])

    # Chọn ra 10 ảnh giống nhất
    top_10_similar_images = similarity_list[:10]

    return top_10_similar_images

#Giao diện Streamlit
st.title("Image Similarity Finder")

# Đường dẫn đến thư mục chứa ảnh đầu vào và thư mục seg
input_path = './seg_test'
seg_path = './seg'

if st.button("Pick Random Image"):
    # Chọn ngẫu nhiên 1 ảnh đầu vào
    input_image_path = pick_random_image(input_path)
    
    # Lấy tên thư mục con
    folder_name = take_folder_name(input_image_path)

    # Tính histogram của các ảnh trong tập seg
    seg_hist_líst = calc_hist_seg_image(seg_path)

    # Tìm 10 ảnh giống nhất
    top_images = find_similar_images(input_image_path, seg_hist_líst)

    # Hiển thị ảnh đầu vào
    st.image(input_image_path, caption=f"Input Image: {folder_name}", width=200)

    st.subheader("Top 10 Similar Images:")
    cols = st.columns(3)  # Tạo các cột cho layout

    # Hien thi 10 ảnh giống nhất    
    # In ra kết quả
    for i, (similarity, img_path) in enumerate(top_images):
        col = cols[i%3]
        with col:
            caption = f"**Output Image**: {take_folder_name(img_path)}<br>**Similarity**: {similarity:.4f}"
            # Sử dụng Markdown để hiển thị tên và độ tương đồng trên hai dòng
            st.image(img_path, use_column_width=True)
            st.markdown(caption, unsafe_allow_html=True)
            
           
        
