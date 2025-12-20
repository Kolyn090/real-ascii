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
from writer import Writer, PositionalCharTemplate  # type: ignore
from arg_util import TraceArgUtil, ShadeArgUtil, ColorArgUtil  # type: ignore
from palette_template import PaletteTemplate  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore

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
    parser.add_argument('--save_chars', action='store_true')
    parser.add_argument('--save_chars_path', type=str, default='./')

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
    t1, ct1, p_ct1 = trace(c1, template, img, args)
    t2, _, p_ct2 = trace(c2, template, img, args)

    converted = t2.copy()
    mask = np.all(ct1[..., :3] < 255, axis=2)
    converted[mask, :3] = t1[mask, :3]

    if args.invert_color:
        converted = invert_image(converted)

    cv2.imwrite(args.save_path, converted)

    gradient_writer = GradientWriter([template], args.max_workers)
    p_cts = gradient_writer._stack([p_ct1, p_ct2])

    if args.save_chars:
        ascii_writer = AsciiWriter(p_cts, int(converted.shape[:2][1]/template.char_bound[0]), args.save_chars_path)
        ascii_writer.save()

    elapsed = time.perf_counter() - start
    print(f"Completed: spent {elapsed:.6f} seconds")

def trace(contour_img: np.ndarray, template: PaletteTemplate,
          original_img: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, list[PositionalCharTemplate]]:
    contour_img = TraceArgUtil.resize(args.resize_method, contour_img, args.resize_factor)
    h, w = contour_img.shape[:2]
    slicer = Slicer(args.max_workers)
    char_bound_width = template.char_bound[0]
    char_bound_height = template.char_bound[1]
    cells = slicer.slice(contour_img, (char_bound_width, char_bound_height))
    writer = template.create_writer(args.max_workers)
    converted, p_cts = writer.match_cells(cells, w, h)
    converted = converted[0:math.floor(h / char_bound_height) * char_bound_height,
                            0:math.floor(w / char_bound_width) * char_bound_width]
    converted_copy = converted.copy()
    original_img = TraceArgUtil.resize(args.resize_method, original_img, args.resize_factor)
    original_img = original_img[0:math.floor(h / char_bound_height) * char_bound_height,
                                0:math.floor(w / char_bound_width) * char_bound_width]

    color_converted = ColorArgUtil.color_image(args.color_option,
                                               converted,
                                               original_img,
                                               (char_bound_width, char_bound_height),
                                               invert_ascii=True)

    if color_converted is not None:
        converted = color_converted
    return converted, converted_copy, p_cts

if __name__ == '__main__':
    main()
