import os
import sys
import cv2
from arg_util import ShadeArgUtil
from static import (to_binary_strong, to_grayscale, increase_contrast,  # type: ignore
                    resize_nearest_neighbor, to_binary_middle, smooth_colors)  # type: ignore
from writer import CharTemplate  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shade')))
from gradient_divide import divide  # type: ignore

def test():
    max_workers = 16
    resize_factor = 4
    contrast_factor = 1
    thresholds_gamma = 0.15
    sigma_s = 1
    sigma_r = 0.6

    image = cv2.imread("../../resource/f_input/ultraman-nexus.png")
    image = resize_nearest_neighbor(image, resize_factor)
    image = increase_contrast(image, contrast_factor)
    image = smooth_colors(image, sigma_s, sigma_r)
    image = to_grayscale(image)

    palettes = ShadeArgUtil.get_palette_json('../../resource/palette_files/jx_files/palette_test.json')
    # palette = palettes[1]
    # flow_writer = palette.create_flow_writer(max_workers)
    # final_img, p_cts = flow_writer.match(image)
    # cv2.imwrite("final_img.png", final_img)

    gradient_imgs = divide(image, len(palettes), thresholds_gamma)
    # os.makedirs("jx_files", exist_ok=True)
    # count = 0
    # for gradient_img in gradient_imgs:
    #     cv2.imwrite(f"jx_files/gradient_{count}.png", gradient_img)
    #     count += 1

    for i in range(len(palettes)):
        palette = palettes[i]
        gradient_img = gradient_imgs[i]
        flow_writer = palette.create_flow_writer(max_workers)
        final_img, p_cts = flow_writer.match(gradient_img)
        cv2.imwrite(f"jx_files/final_{i}.png", final_img)

if __name__ == '__main__':
    test()
