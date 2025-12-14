import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import resize_bilinear, increase_contrast, to_grayscale  # type: ignore

def divide(img_gray: np.ndarray, n_levels: int) -> list[np.ndarray]:
    # Compute gradient magnitude
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # OPTIONAL but highly recommended: ignore flat regions
    valid = grad_mag > 0
    grad_vals = grad_mag[valid]

    # Compute percentile-based thresholds
    bounds = np.percentile(
        grad_vals,
        np.linspace(80, 100, n_levels + 1)
    )

    level_images = []

    for i in range(n_levels):
        lower = bounds[i]
        upper = bounds[i + 1]

        mask = ((grad_mag >= lower) & (grad_mag < upper)).astype(np.uint8) * 255
        level_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        # level_img = np.dstack([img_gray, img_gray, img_gray, mask])
        level_images.append(level_img)

    return level_images

def test():
    factor = 32
    img_path = '../f_input/prof.jpg'
    save_folder = 'test'
    save_to_folder = True
    img = cv2.imread(img_path)
    img = increase_contrast(img, 2)
    img = to_grayscale(img)
    img = resize_bilinear(img, factor)

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    gradient_imgs = divide(img, 8)
    level = 0
    for gradient_img in gradient_imgs:
        save_path = os.path.join(save_folder, f'gradient_{level}.png')
        cv2.imwrite(save_path, gradient_img)
        level += 1

if __name__ == '__main__':
    test()
