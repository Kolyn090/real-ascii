import os
import sys
import cv2
import time
import argparse
import numpy as np

from gradient_writer import GradientWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import (resize_nearest_neighbor, resize_bilinear, invert_image,  # type: ignore
                    floor_fill, increase_contrast, to_grayscale, smooth_colors,  # type: ignore
                    resize_exact)  # type: ignore
from arg_util import ShadeArgUtil, ColorArgUtil, TraceArgUtil  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore
from color_util import (reassign_positional_colors,  # type: ignore
                        process_image_blocks_nonfixed_width, blend_pixels,  # type: ignore
                        average_color_block, blend_ascii_with_color, copy_black_pixels)  # type: ignore
from palette_template import are_palettes_fixed_width, validate_palettes  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../nonfixed_width')))
from nonfixed_width_writer import NonFixedWidthWriter  # type: ignore

def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str, default='ascii_art.png')
    parser.add_argument('--resize_method', type=str, default='nearest_neighbor')
    parser.add_argument('--resize_factor', type=float, default=1)
    parser.add_argument('--contrast_factor', type=float, default=1)
    parser.add_argument('--sigma_s', type=int, default=5)
    parser.add_argument('--sigma_r', type=float, default=0.5)
    parser.add_argument('--thresholds_gamma', type=float, default=0.5)
    parser.add_argument('--palette_path', type=str, default='../../resource/palette_files/palette_default_consolab_fast.json')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--invert_color', action='store_true')
    parser.add_argument('--color_option', type=str, default='')
    parser.add_argument('--save_ascii', action='store_true')
    parser.add_argument('--save_ascii_path', type=str, default='./')
    parser.add_argument('--antialiasing', action='store_true')

    # For non-fixed width
    parser.add_argument('--reference_num', type=int, default=15)
    parser.add_argument('--max_num_fill_item', type=int, default=10)
    parser.add_argument('--filler_lambda', type=float, default=0.7)
    parser.add_argument('--char_weight_sum_factor', type=int, default=50)
    parser.add_argument('--curr_layer_weight_factor', type=int, default=150)
    parser.add_argument('--offset_mse_factor', type=int, default=10)
    parser.add_argument('--coherence_score_factor', type=int, default=5)

    args = parser.parse_args()

    templates = ShadeArgUtil.get_palette_json(args.palette_path)
    img = cv2.imread(args.image_path)
    o_img = img.copy()
    img = TraceArgUtil.resize(args.resize_method, img, args.resize_factor)
    img = increase_contrast(img, args.contrast_factor)
    img = smooth_colors(img, sigma_s=args.sigma_s, sigma_r=args.sigma_r)
    img = to_grayscale(img)

    gradient_writer = GradientWriter(templates, args.max_workers, args.antialiasing)
    gradient_writer.assign_gradient_imgs(img, args.thresholds_gamma)

    palettes = templates
    validate_palettes(palettes)
    are_fixed = are_palettes_fixed_width(palettes)
    if not are_fixed:
        nfww = NonFixedWidthWriter(
            palettes,
            gradient_writer.gradient_imgs,
            args.max_workers,
            reference_num=args.reference_num,
            max_num_fill_item=args.max_num_fill_item,
            filler_lambda=args.filler_lambda,
            char_weight_sum_factor=args.char_weight_sum_factor,
            curr_layer_weight_factor=args.curr_layer_weight_factor,
            offset_mse_factor=args.offset_mse_factor,
            coherence_score_factor=args.coherence_score_factor,
            antialiasing=args.antialiasing
        )
        converted, p_cts = nfww.stack(img.shape[1])
    else:
        converted, p_cts = gradient_writer.match()
    converted = invert_image(converted)

    o_img = resize_exact(converted, o_img)
    large_char_bound = gradient_writer.get_large_char_bound()
    color_result = ColorArgUtil.color_image(args.color_option,
                                            converted,
                                            o_img,
                                            large_char_bound,
                                            antialiasing=args.antialiasing,
                                            invert_ascii=True,
                                            are_fixed=are_fixed,
                                            p_cts=p_cts)
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
                                   int(converted.shape[:2][1]/large_char_bound[0]),
                                   args.save_ascii_path)
        ascii_writer.save()

    elapsed = time.perf_counter() - start
    print(f"Completed: spent {elapsed:.6f} seconds")

if __name__ == '__main__':
    main()
