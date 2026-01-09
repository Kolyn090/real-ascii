import os
import sys
import cv2
import numpy as np

from nonfixed_width_writer import NonFixedWidthWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import (to_binary_strong, to_grayscale, increase_contrast,  # type: ignore
                    resize_nearest_neighbor, to_binary_middle, smooth_colors,  # type: ignore
                    invert_image, resize_exact)  # type: ignore
from arg_util import ShadeArgUtil  # type: ignore
from color_util import (process_image_blocks_nonfixed_width, average_color_block,  # type: ignore
                        blend_ascii_with_color, copy_black_pixels)  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../depth_shade')))
from gradient_divide import divide  # type: ignore

def test():
    os.makedirs("jx_files", exist_ok=True)
    max_workers = 16
    resize_factor = 4
    contrast_factor = 1
    thresholds_gamma = 0.17
    sigma_s = 1
    sigma_r = 0.6

    image = cv2.imread("../../resource/imgs/monalisa.jpg")
    image = resize_nearest_neighbor(image, resize_factor)
    image = increase_contrast(image, contrast_factor)
    image = smooth_colors(image, sigma_s, sigma_r)
    original_img = image.copy()
    image = to_grayscale(image)

    palettes = ShadeArgUtil.get_palette_json('../../resource/palette_files/jx_files/palette_test_padded_6_arial_fast.json')
    gradient_imgs = divide(image, len(palettes), thresholds_gamma)
    nfww = NonFixedWidthWriter(palettes,
                               gradient_imgs,
                               max_workers,
                               reference_num=15,
                               max_num_fill_item=10,
                               filler_lambda=0.7,
                               char_weight_sum_factor=500,
                               curr_layer_weight_factor=150,
                               offset_mse_factor=10,
                               coherence_score_factor=5)

    print(nfww.char_weights)

    # count = 1
    # for img in nfww.transitional_imgs:
    #     cv2.imwrite(f"jx_files/transition_{count}.png", img)
    #     count += 1

    width = resize_factor * image.shape[:2][1]

    converted, p_cts = nfww.stack(width)
    cv2.imwrite(f"jx_files/final_img.png", converted)

    original_img = resize_exact(converted, original_img)

    color_blocks = process_image_blocks_nonfixed_width(original_img, p_cts, average_color_block)[0]
    cv2.imwrite("jx_files/color_blocks.png", color_blocks)

    h = min(color_blocks.shape[0], converted.shape[0])
    w = min(color_blocks.shape[1], converted.shape[1])
    converted = converted[:h, :w]
    color_blocks = color_blocks[:h, :w]

    if converted.ndim == 2:
        converted = np.repeat(converted[:, :, None], 3, axis=2)
    elif converted.ndim == 3 and converted.shape[2] == 1:
        converted = np.repeat(converted, 3, axis=2)

    cv2.imwrite(f"jx_files/converted.png", converted)
    color_converted = blend_ascii_with_color(converted, color_blocks, 1)

    color_converted = copy_black_pixels(converted, color_converted)
    cv2.imwrite(f"jx_files/color_converted.png", color_converted)

    # converted_so = nfww.stack_overlay(width)
    # count = 1
    # for img in converted_so:
    #     cv2.imwrite(f"jx_files/final_img_so_{count}.png", img)
    #     count += 1

if __name__ == '__main__':
    test()
