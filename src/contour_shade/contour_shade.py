import os
import sys
import cv2
import time
import argparse

from eg_writer import EdgeGradientWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from static import invert_image, increase_contrast, to_grayscale, smooth_colors  # type: ignore
from arg_util import ShadeArgUtil, ColorArgUtil, TraceArgUtil  # type: ignore
from ascii_writer import AsciiWriter  # type: ignore
from color_util import reassign_positional_colors  # type: ignore

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
    parser.add_argument('--palette_path', type=str, default='../../resource/palette_files/palette_default.json')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--invert_color', action='store_true')
    parser.add_argument('--sigmaX', type=float, default=0)
    parser.add_argument('--ksize', type=int, default=5)
    parser.add_argument('--gx', type=int, default=3)
    parser.add_argument('--gy', type=int, default=3)
    parser.add_argument('--color_option', type=str, default='')
    parser.add_argument('--save_ascii', action='store_true')
    parser.add_argument('--save_ascii_path', type=str, default='./')

    args = parser.parse_args()

    templates = ShadeArgUtil.get_palette_json(args.palette_path)
    img = cv2.imread(args.image_path)
    img = TraceArgUtil.resize(args.resize_method, img, args.resize_factor)
    o_img = img.copy()
    img = increase_contrast(img, args.contrast_factor)
    img = smooth_colors(img, sigma_s=args.sigma_s, sigma_r=args.sigma_r)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    eg_writer = EdgeGradientWriter(templates, args.max_workers)
    eg_writer.assign_gradient_imgs(img,
                                    args.sigmaX,
                                    args.thresholds_gamma,
                                    args.ksize,
                                    args.gx,
                                    args.gy)
    converted, p_cts = eg_writer.match(w, h)

    large_char_bound = eg_writer.gradient_writer.get_large_char_bound()
    color_result = ColorArgUtil.color_image(args.color_option,
                                            converted,
                                            o_img,
                                            large_char_bound)
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
