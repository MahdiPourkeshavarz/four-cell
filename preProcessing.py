import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    image_filename = 'cells.jpg'

    image_filepath = os.path.join('assets', image_filename)

    original_image = cv2.imread(image_filepath)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
except Exception as e:
    print(f"Error loading image: {e}")
    print("Please make sure you have a file named 'sample_image.jpg' in the same directory.")
    exit()


# Median Blur: Excellent for "salt-and-pepper" noise.
# The second argument is the kernel size (must be an odd number).
median_blurred = cv2.medianBlur(original_image_rgb, 5)

# Gaussian Blur: Smooths the image by averaging pixels with a weighted (Gaussian) kernel.
# The second argument is kernel size, third is standard deviation.
gaussian_blurred = cv2.GaussianBlur(original_image_rgb, (5, 5), 0)



print("Applying CLAHE for contrast amplification...")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)



print("Applying Canny for edge amplification...")
# The second and third arguments are the min and max thresholds for hysteresis.
canny_edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)



plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].imshow(original_image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(median_blurred)
axes[0, 1].set_title('Noise Reduction: Median Blur')
axes[0, 1].axis('off')

axes[0, 2].imshow(gaussian_blurred)
axes[0, 2].set_title('Noise Reduction: Gaussian Blur')
axes[0, 2].axis('off')

axes[1, 0].imshow(gray_image, cmap='gray')
axes[1, 0].set_title('Grayscale Base')
axes[1, 0].axis('off')

axes[1, 1].imshow(clahe_image, cmap='gray')
axes[1, 1].set_title('Contrast Amplification (CLAHE)')
axes[1, 1].axis('off')

axes[1, 2].imshow(canny_edges, cmap='gray')
axes[1, 2].set_title('Edge Amplification (Canny)')
axes[1, 2].axis('off')

plt.suptitle('Image Processing & Amplification with OpenCV', fontsize=20)
plt.tight_layout()
plt.show()

print("\nProcess complete. Displaying results.")