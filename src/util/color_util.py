import math
import os
import sys
from typing import Callable
from static import *
from arg_util import ShadeArgUtil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shade')))
from gradient_writer import GradientWriter  # type: ignore

def blend_ascii_with_color(ascii_img: np.ndarray,
                           color_img: np.ndarray,
                           strength: float) -> np.ndarray:
    ascii_f = ascii_img.astype(np.float32)
    color_f = color_img.astype(np.float32)

    blended = (1 - strength) * color_f + strength * ascii_f

    # Add alpha channel if needed
    if blended.shape[2] == 3:
        alpha = np.full((blended.shape[0], blended.shape[1], 1), 255, dtype=np.float32)
        blended = np.concatenate([blended, alpha], axis=2)

    return np.round(blended).astype(np.uint8)

def copy_black_pixels(source_img: np.ndarray,
                      target_img: np.ndarray) -> np.ndarray:
    assert source_img.shape[:2] == target_img.shape[:2], "Images must have the same dimension."

    result = target_img.copy()
    black_mask = np.all(source_img == 0, axis=2)
    result[black_mask, :3] = source_img[black_mask]
    return result

def process_image_blocks(img: np.ndarray,
                         cell_size: tuple[int, int],
                         block_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    h, w = img.shape[:2]
    cell_w, cell_h = cell_size
    output = img.copy()

    for y in range(0, h, cell_h):
        for x in range(0, w, cell_w):
            block = img[y:y + cell_h, x:x + cell_w]
            processed_block = block_func(block)

            # Ensure processed block has the same size
            output[y:y + cell_h, x:x + cell_w] = processed_block
    output = output[0:math.floor(h / cell_size[1]) * cell_size[1],
                    0:math.floor(w / cell_size[0]) * cell_size[0]]
    return output

def average_color_block(block: np.ndarray) -> np.ndarray:
    avg_color = block.mean(axis=(0, 1), keepdims=True)
    return np.tile(avg_color, (block.shape[0], block.shape[1], 1)).astype(block.dtype)

def test_color():
    resize_factor = 4
    thresholds_gamma = 0.3
    img = cv2.imread('../../resource/imgs/monalisa.jpg')

    color_img = img.copy()
    color_img = resize_nearest_neighbor(color_img, resize_factor)
    cell_size = (13, 22)
    color_converted = process_image_blocks(color_img, cell_size, average_color_block)

    ascii_img = img.copy()
    ascii_img = resize_bilinear(ascii_img, resize_factor)
    ascii_img = smooth_colors(ascii_img, sigma_s=1, sigma_r=0.6)
    ascii_img = to_grayscale(ascii_img)
    h, w = ascii_img.shape[:2]

    templates = ShadeArgUtil.get_palette_json('../../resource/palette_files/palette_default.json')
    gradient_writer = GradientWriter(templates, max_workers=16)
    gradient_writer.assign_gradient_imgs(ascii_img, thresholds_gamma)

    ascii_img = gradient_writer.match(w, h)
    converted = blend_ascii_with_color(ascii_img, color_converted, 0.5)
    converted = copy_black_pixels(ascii_img, converted)
    os.makedirs('test', exist_ok=True)
    cv2.imwrite("test/test.png", converted)

def test_average_color_block():
    resize_factor = 4
    img = cv2.imread('../../resource/imgs/tsunami.jpg')
    img = resize_nearest_neighbor(img, resize_factor)
    cell_size = (13, 22)
    processed = process_image_blocks(img, cell_size, average_color_block)
    os.makedirs('test', exist_ok=True)
    cv2.imwrite("test/test.png", processed)

def test():
    test_color()

if __name__ == '__main__':
    test()
