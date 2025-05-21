import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet

# --- Configuration ---
folder_real = 'dataset/ForenSynths/cyclegan/apple/0_real'  # real images
folder_fake = 'dataset/ForenSynths/cyclegan/apple/1_fake'  # fake images (clean)
folder_noisy = 'dataset/perturbed-data/test/noise/ForenSynths/cyclegan/apple/1_fake'  # fake images (noisy)
output_file = 'fft_comparison_real_fake_noisy.png'
image_size = (256, 256)

def light_wavelet_denoise(img):
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    denoised = denoise_wavelet(img, method='BayesShrink', mode='soft', wavelet_levels=2, channel_axis=None, rescale_sigma=True)
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return denoised

def compute_fft_magnitude(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    log_magnitude = np.log1p(magnitude_spectrum)
    return log_magnitude

def compute_fft_single_image(image_path, image_size, apply_denoise=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, image_size)

    if apply_denoise:
        img = light_wavelet_denoise(img)

    log_magnitude = compute_fft_magnitude(img)

    # Normalize for visualization
    log_magnitude_normalized = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    log_magnitude_normalized = np.uint8(log_magnitude_normalized)

    # Apply color map
    colored_fft = cv2.applyColorMap(log_magnitude_normalized, cv2.COLORMAP_JET)

    return log_magnitude, colored_fft

def compute_abs_diff_img(mag1, mag2):
    diff = np.abs(mag1 - mag2)
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_normalized = np.uint8(diff_normalized)
    diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    return diff_colored

def get_first_image_path(folder, filter_n=False):
    all_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if filter_n:
        all_files = [f for f in all_files if f[:-4].endswith('-n')]
    else:
        all_files = [f for f in all_files if not f[:-4].endswith(('-n', '-b', '-c'))]

    if not all_files:
        raise ValueError(f"No images found in {folder}")

    return os.path.join(folder, all_files[0])

# --- Load first image from each folder ---
image_real = get_first_image_path(folder_real, filter_n=False)
image_fake = get_first_image_path(folder_fake, filter_n=False)
image_noisy = get_first_image_path(folder_noisy, filter_n=True)

print(f"Using real image: {os.path.basename(image_real)}")
print(f"Using fake (clean) image: {os.path.basename(image_fake)}")
print(f"Using fake (noisy) image: {os.path.basename(image_noisy)}")

# --- Compute FFTs ---
log_fft_real, fft_real = compute_fft_single_image(image_real, image_size, apply_denoise=False)
log_fft_fake, fft_fake = compute_fft_single_image(image_fake, image_size, apply_denoise=False)
log_fft_noisy, fft_noisy = compute_fft_single_image(image_noisy, image_size, apply_denoise=False)

log_fft_real_denoised, fft_real_denoised = compute_fft_single_image(image_real, image_size, apply_denoise=True)
log_fft_fake_denoised, fft_fake_denoised = compute_fft_single_image(image_fake, image_size, apply_denoise=True)
log_fft_noisy_denoised, fft_noisy_denoised = compute_fft_single_image(image_noisy, image_size, apply_denoise=True)

# --- Compute differences ---
# Row 3: Diff original vs denoised
diff_real = compute_abs_diff_img(log_fft_real, log_fft_real_denoised)
diff_fake = compute_abs_diff_img(log_fft_fake, log_fft_fake_denoised)
diff_noisy = compute_abs_diff_img(log_fft_noisy, log_fft_noisy_denoised)

# Row 4: Diff fake clean vs fake noisy
diff_fake_vs_noisy = compute_abs_diff_img(log_fft_fake, log_fft_noisy)
diff_fake_vs_noisy_denoised = compute_abs_diff_img(log_fft_fake_denoised, log_fft_noisy_denoised)

# Row 5: Diff fake clean (original) vs fake noisy (denoised)
diff_fake_clean_vs_noisy_denoised = compute_abs_diff_img(log_fft_fake, log_fft_noisy_denoised)

# --- Save combined output (optional: cv2 image) ---
combined = np.vstack([
    np.hstack((fft_real, fft_fake, fft_noisy)),                         # Row 1
    np.hstack((fft_real_denoised, fft_fake_denoised, fft_noisy_denoised)),  # Row 2
    np.hstack((diff_real, diff_fake, diff_noisy)),                      # Row 3
    np.hstack((np.zeros_like(diff_real), diff_fake_vs_noisy, diff_fake_vs_noisy_denoised)),  # Row 4
    np.hstack((np.zeros_like(diff_real), np.zeros_like(diff_real), diff_fake_clean_vs_noisy_denoised))  # Row 5
])
cv2.imwrite(output_file, combined)
print(f"Saved comparison image to {output_file}")

# --- Plot nicely with Matplotlib ---
plt.figure(figsize=(18, 30))

# Row 1 - Original FFTs
plt.subplot(5, 3, 1)
plt.imshow(cv2.cvtColor(fft_real, cv2.COLOR_BGR2RGB))
plt.title('Real Apple (Original)')
plt.axis('off')

plt.subplot(5, 3, 2)
plt.imshow(cv2.cvtColor(fft_fake, cv2.COLOR_BGR2RGB))
plt.title('Fake Apple (Clean, Original)')
plt.axis('off')

plt.subplot(5, 3, 3)
plt.imshow(cv2.cvtColor(fft_noisy, cv2.COLOR_BGR2RGB))
plt.title('Fake Apple (Noisy, Original)')
plt.axis('off')

# Row 2 - Denoised FFTs
plt.subplot(5, 3, 4)
plt.imshow(cv2.cvtColor(fft_real_denoised, cv2.COLOR_BGR2RGB))
plt.title('Real Apple (Denoised)')
plt.axis('off')

plt.subplot(5, 3, 5)
plt.imshow(cv2.cvtColor(fft_fake_denoised, cv2.COLOR_BGR2RGB))
plt.title('Fake Apple (Clean, Denoised)')
plt.axis('off')

plt.subplot(5, 3, 6)
plt.imshow(cv2.cvtColor(fft_noisy_denoised, cv2.COLOR_BGR2RGB))
plt.title('Fake Apple (Noisy, Denoised)')
plt.axis('off')

# Row 3 - Diff Original vs Denoised
plt.subplot(5, 3, 7)
plt.imshow(cv2.cvtColor(diff_real, cv2.COLOR_BGR2RGB))
plt.title('Real (Original vs Denoised)')
plt.axis('off')

plt.subplot(5, 3, 8)
plt.imshow(cv2.cvtColor(diff_fake, cv2.COLOR_BGR2RGB))
plt.title('Fake Clean (Original vs Denoised)')
plt.axis('off')

plt.subplot(5, 3, 9)
plt.imshow(cv2.cvtColor(diff_noisy, cv2.COLOR_BGR2RGB))
plt.title('Fake Noisy (Original vs Denoised)')
plt.axis('off')

# Row 4 - Diff Fake Clean vs Fake Noisy
plt.subplot(5, 3, 10)
plt.imshow(cv2.cvtColor(diff_fake_vs_noisy, cv2.COLOR_BGR2RGB))
plt.title('Fake Clean vs Fake Noisy (Original)')
plt.axis('off')

plt.subplot(5, 3, 11)
plt.imshow(cv2.cvtColor(diff_fake_vs_noisy_denoised, cv2.COLOR_BGR2RGB))
plt.title('Fake Clean vs Fake Noisy (Both Denoised)')
plt.axis('off')

# Row 5 - Diff Fake Clean (original) vs Fake Noisy (after denoising)
plt.subplot(5, 3, 12)
plt.imshow(cv2.cvtColor(diff_fake_clean_vs_noisy_denoised, cv2.COLOR_BGR2RGB))
plt.title('Fake Clean (Original) vs Fake Noisy (Denoised)')
plt.axis('off')

plt.tight_layout()
plt.show()
