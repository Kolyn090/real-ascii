import os.path
import sys
import cv2

from gradient_writer import GradientWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from static import (resize_nearest_neighbor, resize_bilinear, invert_image,   # type: ignore
                    floor_fill, increase_contrast, to_grayscale, smooth_colors)  # type: ignore

def main():
    factor = 4
    img_path = '../f_input/prof.jpg'
    save_folder = 'test_writer'
    save_to_folder = True
    templates = [
        [" "],
        [" ", ".", ",", "-", "_", "^", "'"],
        [" ", ":", ";", "!", "i"],
        ["+", "=", "*", "l", "[", "]", "~"],
        ["%", "&", "8", "B"],
        ["W", "M", "@", "$", "#"]
    ]
    img = cv2.imread(img_path)
    img = increase_contrast(img, 2)
    if save_folder:
        cv2.imwrite(os.path.join(save_folder, "original_img.png"), img)

    img = resize_bilinear(img, factor)
    img = smooth_colors(img, sigma_s=1, sigma_r=0.6)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    if save_folder:
        cv2.imwrite(os.path.join(save_folder, "img.png"), img)

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    gradient_writer = GradientWriter()
    gradient_writer.save_to_folder = save_to_folder
    gradient_writer.save_folder = save_folder
    gradient_writer.templates = templates
    gradient_writer.assign_gradient_imgs(img, 0.3)

    for i in range(len(gradient_writer.gradient_imgs)):
        if save_to_folder:
            cv2.imwrite(os.path.join(save_folder, f"gradient_{i}.png"), gradient_writer.gradient_imgs[i])

    gradient_writer.match(w, h)

if __name__ == '__main__':
    main()
