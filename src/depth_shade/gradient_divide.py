import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import (resize_bilinear, increase_contrast,  # type: ignore
                    to_grayscale, smooth_colors)  # type: ignore

def divide(img_gray: np.ndarray, n_levels: int, thresholds_gamma: float) -> list[np.ndarray]:
    grad_scaled = img_gray / img_gray.max() * 255.0
    thresholds = compute_equal_pixel_thresholds(img_gray, n_levels, thresholds_gamma)
    print(f"Gradient Thresholds: {thresholds}")

    level_images = []

    for i in range(n_levels):
        lower = thresholds[i]
        upper = thresholds[i + 1]

        # Mask in scaled gradient
        mask = ((grad_scaled >= lower) & (grad_scaled <= upper)).astype(np.uint8)
        level_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        level_img[level_img > 0] = 255
        level_images.append(level_img)

    return level_images

def compute_equal_pixel_thresholds(img_gray: np.ndarray,
                                   n_levels: int,
                                   gamma: float = 1.0) -> list[int]:
    """
        Compute thresholds for a grayscale image such that each level has roughly equal number of pixels.

        Args:
            img_gray: 2D np.ndarray of dtype uint8 (0-255)
            n_levels: Number of desired levels
            gamma: Optional float >0 to bias thresholds toward bright pixels (>1 emphasizes bright pixels)

        Returns:
            List of n_levels+1 integers representing thresholds [t0, t1, ..., t_n_levels]
            Pixels between thresholds[i] and thresholds[i+1] belong to level i.
        """
    # 1. Compute histogram
    hist = np.bincount(img_gray.flatten(), minlength=256)
    total_pixels = hist.sum()

    # 2. Compute cumulative histogram (CDF)
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / total_pixels  # range 0..1

    # 3. Determine target fractions for each threshold
    fractions = np.linspace(0, 1, n_levels + 1)  # linear
    if gamma != 1.0:
        fractions = fractions ** gamma  # bias toward bright pixels

    # 4. Map fractions to gray levels using CDF
    thresholds = [0]  # always include 0
    for f in fractions[1:-1]:  # skip first (0) and last (1)
        gray_val = np.searchsorted(cdf_normalized, f)
        thresholds.append(int(gray_val))
    thresholds.append(255)  # always include max

    return thresholds

def test():
    factor = 20
    img_path = '../../resource/f_input/ultraman-nexus.png'
    save_folder = 'test'
    save_to_folder = True
    img = cv2.imread(img_path)
    img = increase_contrast(img, 2)
    img = to_grayscale(img)
    img = resize_bilinear(img, factor)
    img = smooth_colors(img, sigma_s=50, sigma_r=0.05)

    cv2.imwrite("test.png", img)

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    gradient_imgs = divide(img, 4, 0.3)
    level = 0
    for gradient_img in gradient_imgs:
        save_path = os.path.join(save_folder, f'gradient_{level}.png')
        cv2.imwrite(save_path, gradient_img)
        level += 1

if __name__ == '__main__':
    test()
