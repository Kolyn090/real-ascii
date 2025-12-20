import os
import sys
import time

import cv2
import math
import argparse
import numpy as np
from PIL import ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from arg_util import TraceArgUtil, ShadeArgUtil, ColorArgUtil  # type: ignore
from palette_template import PaletteTemplate  # type: ignore
from static import invert_image  # type: ignore
# from color_util import copy_non_black_pixels_to_white  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore

def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str, default='ascii_art.png')
    parser.add_argument('--resize_factor', type=float, default=1)
    parser.add_argument('--invert_color', action='store_true')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--resize_method', type=str, default='nearest_neighbor')

    parser.add_argument('--chars', type=str, default='')
    parser.add_argument('--chars_file_path', type=str, default='../../resource/char_files/chars_file.txt')
    parser.add_argument('--font', type=str, default='')
    parser.add_argument('--font_size', type=int, default=24)
    parser.add_argument('--char_bound_width', type=int, default=-1)
    parser.add_argument('--char_bound_height', type=int, default=-1)
    parser.add_argument('--match_method', type=str, default='')
    parser.add_argument('--approx_ratio', type=float, default=-1)
    parser.add_argument('--vector_top_k', type=int, default=-1)
    parser.add_argument('--color_option', type=str, default='')
    parser.add_argument('--original_image_path', type=str, default='')

    # Including, can be overridden with explicit arguments:
    # chars
    # font
    # font_size
    # char_bound_width
    # char_bound_height
    # approx_ratio
    # vector_top_k
    # match_method
    parser.add_argument('--palette_path', type=str, default='')
    parser.add_argument('--save_chars', action='store_true')
    parser.add_argument('--save_chars_path', type=str, default='./')

    args = parser.parse_args()
    template = assemble_template(args)

    img = cv2.imread(args.image_path)
    img = TraceArgUtil.resize(args.resize_method, img, args.resize_factor)
    h, w = img.shape[:2]

    slicer = Slicer(args.max_workers)
    char_bound_width = template.char_bound[0]
    char_bound_height = template.char_bound[1]
    cells = slicer.slice(img, (char_bound_width, char_bound_height))
    writer = template.create_writer(args.max_workers)
    converted, p_cts = writer.match_cells(cells, w, h)
    converted = converted[0:math.floor(h / char_bound_height) * char_bound_height,
                            0:math.floor(w / char_bound_width) * char_bound_width]

    original_img = get_original_image(args)
    original_img = original_img[0:math.floor(h / char_bound_height) * char_bound_height,
                                0:math.floor(w / char_bound_width) * char_bound_width]

    color_converted = ColorArgUtil.color_image(args.color_option,
                                               converted,
                                               original_img,
                                               (char_bound_width, char_bound_height),
                                               invert_ascii=True)

    if color_converted is not None:
        # color_converted = copy_non_black_pixels_to_white(converted, color_converted)
        converted = color_converted

    if args.invert_color:
        converted = invert_image(converted)
    cv2.imwrite(args.save_path, converted)

    if args.save_chars:
        ascii_writer = AsciiWriter(p_cts, int(converted.shape[:2][1]/char_bound_width), args.save_chars_path)
        ascii_writer.save()

    elapsed = time.perf_counter() - start
    print(f"Completed: spent {elapsed:.6f} seconds")

def assemble_template(args) -> PaletteTemplate | None:
    template: PaletteTemplate = None
    if os.path.exists(args.palette_path):
        template = ShadeArgUtil.get_palette_json(args.palette_path)[0]

    if template is None:
        template = PaletteTemplate(
            layer=0,
            chars=TraceArgUtil.get_chars(args.chars, args.chars_file_path),
            imageFont=ImageFont.truetype(args.font, args.font_size),
            char_bound=(args.char_bound_width, args.char_bound_height),
            approx_ratio=args.approx_ratio,
            vector_top_k=args.vector_top_k,
            match_method=args.match_method
        )
        return template

    # Override chars, if possible
    if args.chars != '':
        chars = TraceArgUtil.get_chars(args.chars, args.chars_file_path)
        template.chars = chars

    # Override font, if possible
    if args.font != '':
        font = ImageFont.truetype(args.font, args.font_size)
        template.imageFont = font

    # Override char_bound_width and height, if possible
    if args.char_bound_width != -1 and args.char_bound_height != -1:
        template.char_bound = (args.char_bound_width, args.char_bound_height)

    # Override approx_ratio, if possible
    if args.approx_ratio != -1:
        template.approx_ratio = args.approx_ratio

    # Override vector_top_k, if possible
    if args.vector_top_k != -1:
        template.vector_top_k = args.vector_top_k

    # Override match method, if possible
    if args.match_method != '':
        template.match_method = args.match_method

    return template

def get_original_image(args) -> np.ndarray | None:
    original_img_path = args.original_image_path
    if not os.path.exists(original_img_path):
        return None
    img = cv2.imread(original_img_path)
    img = TraceArgUtil.resize(args.resize_method, img, args.resize_factor)
    return img

if __name__ == '__main__':
    main()
