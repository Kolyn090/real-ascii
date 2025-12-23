import os
import sys
import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from row_img_reconstruct import reconstruct, reconstruct2  # type: ignore
from static import (to_binary_strong, to_grayscale, increase_contrast,  # type: ignore
                    resize_nearest_neighbor, to_binary_middle)  # type: ignore
from writer import CharTemplate  # type: ignore

def test_from_complete_example():
    gap = 1
    char_bound_height = 28
    resize_factor = 8
    contrast_factor = 1
    font_path = "C:/Windows/Fonts/arial.ttf"
    font_size = 24
    chars = {
        " ": (6, char_bound_height),
        "a": (11, 12),
        "b": (11, 18),
        "c": (11, 12),
        "\\": (11, 20),
        "/": (11, 20),
        "-": (8, 2),
        "#": (14, 16),
        "$": (11, 21),
        "@": (22, 19),
        "^": (12, 10)
    }
    # Uniform height
    chars = {key: (val[0]+2*gap, char_bound_height) for key, val in chars.items()}

    image = cv2.imread("../../../resource/f_input/ultraman-nexus.png")
    image = resize_nearest_neighbor(image, resize_factor)
    image = increase_contrast(image, contrast_factor)
    image = to_grayscale(image)
    image = to_binary_middle(image, 90)
    cv2.imwrite("jx_files/binary.png", image)
    long_imgs = split_into_rows(image, char_bound_height)

    # Verify long images
    # os.makedirs("jx_files/long_imgs", exist_ok=True)
    # count = 0
    # for long_img in long_imgs:
    #     cv2.imwrite(f"jx_files/long_imgs/long_img_{count}.png", long_img)
    #     count += 1

    # Make the font
    image_font = ImageFont.truetype(font_path, font_size)
    char_templates = [create_char_template(char, char_bound, image_font) for char, char_bound in chars.items()]
    short_imgs = [char_template.img for char_template in char_templates]

    # Verify short images
    os.makedirs("jx_files/short_imgs", exist_ok=True)
    count = 0
    for short_img in short_imgs:
        cv2.imwrite(f"jx_files/short_imgs/short_img_{count}.png", short_img)
        count += 1

    rows = []
    for long_img in long_imgs:
        reconstructed = reconstruct2(long_img, char_templates)
        seq, score = reconstructed
        seq = [ct.img for ct in seq]
        rows.append(concat_images_left_to_right(seq))
    final_img = concat_images_top_to_bottom(rows, pad_color=(255, 255, 255))
    cv2.imwrite("jx_files/final.png", final_img)

def create_char_template(char: str,
                         char_bound: tuple[int, int],
                         image_font: FreeTypeFont) -> CharTemplate:
    img = Image.new("RGB", char_bound, "white")
    draw = ImageDraw.Draw(img)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), char, font=image_font)

    text_w = bbox[2] - bbox[0]
    # text_h = bbox[3] - bbox[1]

    img_w, img_h = img.size

    # Center position
    x = (img_w - text_w) // 2
    # y = (img_h - text_h) // 2 - bbox[1]

    draw.text((x, 0), char, font=image_font, fill="black")

    template = np.array(img)
    template_binary = to_binary_strong(template)
    template_small = template_binary
    template_small = to_binary_strong(template_small)
    char_template = CharTemplate(
        char=char,
        image_font=image_font,
        char_bound=char_bound,
        img=template,
        img_binary=template_binary,
        img_small=template_small,
        img_projection=template_small.ravel()
    )
    return char_template

def split_into_rows(img: np.ndarray, row_height: int) -> list[np.ndarray]:
    H = img.shape[0]
    rows = []
    for y in range(0, H - row_height + 1, row_height):
        rows.append(img[y:y + row_height])
    return rows

def concat_images_left_to_right(images: list[np.ndarray]) -> np.ndarray:
    """
    Concatenates a list of images (all same height) horizontally (left to right).
    """
    # check that all images have the same height
    heights = [img.shape[0] for img in images]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height")

    # concatenate along the width axis
    concatenated = np.hstack(images)
    return concatenated

def concat_images_top_to_bottom(images: list[np.ndarray], pad_color=(0, 0, 0)) -> np.ndarray:
    """
    Concatenates images vertically. If widths differ, pad them to the max width.

    pad_color: tuple or scalar
        Color used for padding (RGB for color images, single int for grayscale)
    """
    if not images:
        raise ValueError("Image list is empty")

    # Determine max width and height per image
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_width = max(widths)

    padded_images = []
    for img in images:
        h, w = img.shape[:2]

        if w == max_width:
            padded_images.append(img)
            continue

        # Create padded canvas
        if img.ndim == 3:  # color image
            pad_img = np.full((h, max_width, img.shape[2]), pad_color, dtype=img.dtype)
        else:  # grayscale
            pad_img = np.full((h, max_width), pad_color, dtype=img.dtype)

        # place image on left, pad on right
        pad_img[:, :w] = img
        padded_images.append(pad_img)

    return np.vstack(padded_images)

# def find_all_png(root: str) -> list[str]:
#     root = Path(root)
#     png_files = list(root.rglob("*.png"))
#     return [str(png_file) for png_file in png_files]

if __name__ == '__main__':
    test_from_complete_example()
