import math
import os
import sys
import time

import cv2
import argparse
import numpy as np

from contour import contour
from trace import assemble_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import increase_contrast, invert_image  # type: ignore
from slicer import Slicer  # type: ignore
from char_template import PositionalCharTemplate  # type: ignore
from writer import Writer  # type: ignore
from arg_util import TraceArgUtil, ShadeArgUtil, ColorArgUtil  # type: ignore
from palette_template import PaletteTemplate  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore
from color_util import PositionalColor, reassign_positional_colors  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shade')))
from gradient_writer import GradientWriter  # type: ignore

def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--canny1', type=int, default=0)
    parser.add_argument('--canny2', type=int, default=0)
    parser.add_argument('--gb_size', type=int, default=5)
    parser.add_argument('--gb_sigmaX', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--dilate_iter', type=int, default=1)
    parser.add_argument('--erode_iter', type=int, default=1)
    parser.add_argument('--contrast_factor', type=float, default=1)
    parser.add_argument('--contrast_window_size', type=int, default=8)

    parser.add_argument('--save_path', type=str, default='ascii_art.png')
    parser.add_argument('--resize_factor', type=float, default=1)
    parser.add_argument('--resize_method', type=str, default='nearest_neighbor')
    parser.add_argument('--max_workers', type=int, default=16)

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
    parser.add_argument('--palette_path', type=str, default='')

    parser.add_argument('--invert_color', action='store_true')
    parser.add_argument('--save_ascii', action='store_true')
    parser.add_argument('--save_ascii_path', type=str, default='./')

    args = parser.parse_args()
    template = assemble_template(args)
    img = cv2.imread(args.image_path)
    c1 = contour(img, args.canny1, args.canny2,
                gb_size=args.gb_size,
                gb_sigmaX=args.gb_sigmaX,
                kernel_size=args.kernel_size,
                dilate_iter=args.dilate_iter,
                erode_iter=args.erode_iter,
                contrast_factor=args.contrast_factor,
                contrast_window_size=args.contrast_window_size,
                invert_color=False)
    c2 = contour(img, args.canny1, args.canny2,
                gb_size=args.gb_size,
                gb_sigmaX=args.gb_sigmaX,
                kernel_size=args.kernel_size,
                dilate_iter=args.dilate_iter,
                erode_iter=args.erode_iter,
                contrast_factor=args.contrast_factor,
                contrast_window_size=args.contrast_window_size,
                invert_color=True)

    p_cs, color_blocks, stacked, converted = trace_join(c1, c2, template, img, args)

    if args.invert_color:
        if color_blocks is not None:
            color_blocks = invert_image(color_blocks)
        converted = invert_image(converted)

    cv2.imwrite(args.save_path, converted)

    if args.save_ascii:
        reassign_positional_colors(p_cs, color_blocks)
        ascii_writer = AsciiWriter(stacked,
                                   p_cs,
                                   int(converted.shape[:2][1]/template.char_bound[0]),
                                   args.save_ascii_path)
        ascii_writer.save()

    elapsed = time.perf_counter() - start
    print(f"Completed: spent {elapsed:.6f} seconds")

def trace_join(contour1: np.ndarray, contour2: np.ndarray,
               template: PaletteTemplate, original_img: np.ndarray, args):
    contour1 = TraceArgUtil.resize(args.resize_method, contour1, args.resize_factor)
    contour2 = TraceArgUtil.resize(args.resize_method, contour2, args.resize_factor)
    h, w = contour1.shape[:2]

    slicer = Slicer(args.max_workers)
    char_bound_width = template.char_bound[0]
    char_bound_height = template.char_bound[1]
    cells1 = slicer.slice(contour1, (char_bound_width, char_bound_height))
    cells2 = slicer.slice(contour2, (char_bound_width, char_bound_height))

    writer = template.create_writer(args.max_workers)
    converted1, p_cts1 = writer.match_cells(cells1, w, h)
    converted1 = converted1[0:math.floor(h / char_bound_height) * char_bound_height,
                            0:math.floor(w / char_bound_width) * char_bound_width]
    converted2, p_cts2 = writer.match_cells(cells2, w, h)

    original_img = TraceArgUtil.resize(args.resize_method, original_img, args.resize_factor)
    original_img = original_img[0:math.floor(h / char_bound_height) * char_bound_height,
                                0:math.floor(w / char_bound_width) * char_bound_width]
    converted1 = invert_image(converted1)
    color_result1 = ColorArgUtil.color_image(args.color_option,
                                             converted1,
                                             original_img,
                                             (char_bound_width, char_bound_height))
    color_blocks1 = None
    p_cs1 = []
    if color_result1 is not None:
        _, color_blocks1, p_cs1 = color_result1

    # Current goal: place ascii 1 on top of ascii 2
    stacked = stack_ascii_p_cts(p_cts2, p_cts1)

    # To be returned
    p_cs = p_cs1
    color_blocks = color_blocks1

    gradient_writer = GradientWriter([template], args.max_workers)
    converted = gradient_writer.stack_to_img(stacked, w, h)

    color_result = ColorArgUtil.color_image(args.color_option,
                                               converted,
                                               original_img,
                                               (char_bound_width, char_bound_height))

    if color_result is not None:
        converted, _, _ = color_result

    return p_cs, color_blocks, stacked, converted

def stack_ascii_p_cts(bottom: list[PositionalCharTemplate], top: list[PositionalCharTemplate]) \
        -> list[PositionalCharTemplate]:
    table: dict[tuple[int, int], PositionalCharTemplate] = dict()
    table = {p_ct.top_left: p_ct for p_ct in bottom}
    for p_ct in top:
        if p_ct.char_template.char != ' ':
            table[p_ct.top_left] = p_ct
    return list(table.values())

if __name__ == '__main__':
    main()
