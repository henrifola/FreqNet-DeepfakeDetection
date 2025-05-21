import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True, help='Input folder path')
parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')
parser.add_argument('--h', type=int, default=3, help='Denoising strength h (default=8)')
parser.add_argument('--template', type=int, default=3, help='Template window size (default=10)')
parser.add_argument('--search', type=int, default=25, help='Search window size (default=3)')
args = parser.parse_args()
# H=8, TEMPLATE=10, SEARCH=2
BEST_H = args.h
BEST_TEMPLATE = args.template
BEST_SEARCH = args.search
image_size = (256, 256)  # Fixed resize if needed

# --- Denoising function ---
def light_denoise(img):
    img = img.astype(np.uint8)
    channels = cv2.split(img)
    denoised_channels = []
    for ch in channels:
        denoised_ch = cv2.fastNlMeansDenoising(
            ch, h=BEST_H,
            templateWindowSize=BEST_TEMPLATE,
            searchWindowSize=BEST_SEARCH
        )
        denoised_channels.append(denoised_ch)
    denoised = cv2.merge(denoised_channels)
    return denoised

# --- Safe image read ---
def safe_imread(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# --- Process recursively ---
def process_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    items = os.listdir(src_folder)
    for item in tqdm(items, desc=f"Processing {src_folder}"):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isdir(src_path):
            process_folder(src_path, dst_path)
        elif os.path.isfile(src_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = safe_imread(src_path)
            if img is not None:
                # img = cv2.resize(img, image_size)
                denoised = light_denoise(img)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                cv2.imwrite(dst_path, denoised)

if __name__ == "__main__":
    print(f"Starting denoising:\nFrom: {args.input_folder}\nTo: {args.output_folder}\nH={BEST_H}, Template={BEST_TEMPLATE}, Search={BEST_SEARCH}")
    process_folder(args.input_folder, args.output_folder)
    print("✅ All images denoised and saved (color preserved).")
