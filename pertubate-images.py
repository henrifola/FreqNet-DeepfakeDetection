import os
import shutil
import argparse
import cv2
import random
import numpy as np

distortion_count = 0
blur_count = 0
compress_count = 0
noise_count = 0

def safe_imread(filepath):
    print(f"Reading image from: {filepath}")
    try:
        file_bytes = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except:
        return None

def perturb_image(img):
    global distortion_count, blur_count, compress_count, noise_count

    if img is None:
        raise ValueError("Received None image for perturbation.")

    if random.random() < 0.5:
        filter_choice = random.choice(['blur', 'compress', 'noise'])
        print(f"Applying {filter_choice} perturbation")

        if filter_choice == 'blur':
            ksize = random.choice([3, 5, 7, 9])
            img_out = cv2.GaussianBlur(img, (ksize, ksize), 0)
            if img_out is None:
                raise ValueError("Blur perturbation failed.")
            blur_count += 1
            distortion_count += 1
            return img_out, 'b'

        elif filter_choice == 'compress':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(10, 75)]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if not result:
                raise ValueError("Compression encode failed.")
            img_out = cv2.imdecode(encimg, 1)
            if img_out is None:
                raise ValueError("Compression decode failed.")
            compress_count += 1
            distortion_count += 1
            return img_out, 'c'

        elif filter_choice == 'noise':
            img_out = img.astype(np.float32)
            variance = random.uniform(5.0, 20.0)
            stddev = variance ** 0.5
            noise = np.random.normal(0, stddev, img.shape).astype(np.float32)
            img_out += noise
            img_out = np.clip(img_out, 0, 255).astype(np.uint8)
            if img_out is None:
                raise ValueError("Noise perturbation failed.")
            noise_count += 1
            distortion_count += 1
            return img_out, 'n'

        else:
            raise ValueError(f"Unknown filter choice: {filter_choice}")

    else:
        return img, None

def make_file(item, dst_folder_path, src_folder_path):
    src_file_path = os.path.join(src_folder_path, item)
    dst_file_path = os.path.join(dst_folder_path, item)

    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

    if not os.path.exists(dst_file_path):
        img = safe_imread(src_file_path)
        if img is None:
            raise RuntimeError(f"Error reading image: {src_file_path}")

        perturbed_img, perturbation_type = perturb_image(img)
        if perturbed_img is None:
            raise RuntimeError(f"Error perturbing image: {src_file_path}")

        if perturbation_type is not None:
            base, ext = os.path.splitext(dst_file_path)
            dst_file_path = f"{base}-{perturbation_type}{ext}"

        cv2.imwrite(dst_file_path, perturbed_img)
    else:
        print(f"Skipped {item}, already exists at {dst_file_path}")

def process_folder(src_folder, dst_folder):
    file_count = 0
    print(f"Processing folder: {src_folder} -> {dst_folder}")
    os.makedirs(dst_folder, exist_ok=True)

    for item in os.listdir(src_folder):
        src_item_path = os.path.join(src_folder, item)
        dst_item_path = os.path.join(dst_folder, item)
        if os.path.isfile(src_item_path):
            make_file(item, dst_folder, src_folder)
            file_count += 1
        elif os.path.isdir(src_item_path):
            file_count += process_folder(src_item_path, dst_item_path)
        else:
            pass
            
    return file_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_base")
    parser.add_argument("dst_base")
    args = parser.parse_args()

    src_base = args.src_base
    dst_base = args.dst_base

    os.makedirs(dst_base, exist_ok=True)
    total_files = process_folder(src_base, dst_base)

    print(f"Total files processed: {total_files}")
    print(f"Total files where distortion was applied: {distortion_count}")
    print(f"Blur applied: {blur_count}")
    print(f"Compression applied: {compress_count}")
    print(f"Noise applied: {noise_count}")
    print(f"Files copied from {src_base} to {dst_base}")
    print("Copying completed.")
