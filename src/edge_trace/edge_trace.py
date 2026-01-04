import os
import sys
import time

import cv2
import argparse
import numpy as np
from PIL import ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from arg_util import TraceArgUtil, ShadeArgUtil, ColorArgUtil  # type: ignore
from palette_template import PaletteTemplate, are_palettes_fixed_width  # type: ignore
from static import invert_image, resize_exact  # type: ignore
from color_util import reassign_positional_colors  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../nonfixed_width')))
from nonfixed_width_writer import NonFixedWidthWriter  # type: ignore

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
    parser.add_argument('--pad_width', type=int, default=0)
    parser.add_argument('--pad_height', type=int, default=0)
    parser.add_argument('--antialiasing', action='store_true')

    # For non-fixed width
    parser.add_argument('--reference_num', type=int, default=15)
    parser.add_argument('--max_num_fill_item', type=int, default=10)
    parser.add_argument('--filler_lambda', type=float, default=0.7)
    parser.add_argument('--char_weight_sum_factor', type=int, default=50)
    parser.add_argument('--curr_layer_weight_factor', type=int, default=150)
    parser.add_argument('--offset_mse_factor', type=int, default=10)
    parser.add_argument('--coherence_score_factor', type=int, default=5)

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
    parser.add_argument('--save_ascii', action='store_true')
    parser.add_argument('--save_ascii_path', type=str, default='./')
    parser.add_argument('--antialiasing', action='store_true')

    args = parser.parse_args()
    template = assemble_template(args)

    img = cv2.imread(args.image_path)
    img = TraceArgUtil.resize(args.resize_method, img, args.resize_factor)

    are_fixed = are_palettes_fixed_width([template])
    if not are_fixed:
        nfww = NonFixedWidthWriter([template],
                                   [img],
                                   args.max_workers,
                                   reference_num=args.reference_num,
                                   max_num_fill_item=args.max_num_fill_item,
                                   filler_lambda=args.filler_lambda,
                                   char_weight_sum_factor=args.char_weight_sum_factor,
                                   curr_layer_weight_factor=args.curr_layer_weight_factor,
                                   offset_mse_factor=args.offset_mse_factor,
                                   coherence_score_factor=args.coherence_score_factor)
        converted, p_cts = nfww.stack(img.shape[1])
    else:
        slicer = Slicer(args.max_workers)
        padded_char_bound = (template.char_bound[0] + 2*template.pad[0], template.char_bound[1] + 2*template.pad[1])
        cells = slicer.slice(img, padded_char_bound)
        writer = template.create_writer(args.max_workers, args.antialiasing)
        converted, p_cts = writer.match_cells(cells)

    original_img = get_original_image(args)
    if original_img is not None:
        original_img = resize_exact(converted, original_img)

    color_result = ColorArgUtil.color_image(args.color_option,
                                            converted,
                                            original_img,
                                            template.char_bound,
                                            antialiasing=args.antialiasing,
                                            invert_ascii=are_fixed)
    color_blocks = None
    p_cs = []
    if color_result is not None:
        color_converted, color_blocks, p_cs = color_result
        converted = color_converted

    if args.invert_color:
        if color_blocks is not None:
            color_blocks = invert_image(color_blocks)
        converted = invert_image(converted)

    cv2.imwrite(args.save_path, converted)

    if args.save_ascii:
        reassign_positional_colors(p_cs, color_blocks)
        ascii_writer = AsciiWriter(p_cts,
                                   p_cs,
                                   int(converted.shape[:2][1]/template.char_bound[0]),
                                   args.save_ascii_path)
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
            image_font=ImageFont.truetype(args.font, args.font_size),
            char_bound=(args.char_bound_width, args.char_bound_height),
            approx_ratio=args.approx_ratio,
            vector_top_k=args.vector_top_k,
            match_method=args.match_method,
            pad=(args.pad_width, args.pad_height)
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
    return cv2.imread(original_img_path)

if __name__ == '__main__':
    main()
