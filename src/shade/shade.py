import os.path
import sys
import time

import cv2
import argparse

from gradient_writer import GradientWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from static import (resize_nearest_neighbor, resize_bilinear, invert_image,   # type: ignore
                    floor_fill, increase_contrast, to_grayscale, smooth_colors)  # type: ignore
from arg_util import ShadeArgUtil  # type: ignore

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

    args = parser.parse_args()

    templates = ShadeArgUtil.get_palette_json(args.palette_path)
    img = cv2.imread(args.image_path)
    img = increase_contrast(img, args.contrast_factor)
    img = resize_bilinear(img, args.resize_factor)
    img = smooth_colors(img, sigma_s=args.sigma_s, sigma_r=args.sigma_r)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    gradient_writer = GradientWriter(templates, args.max_workers)
    gradient_writer.assign_gradient_imgs(img, args.thresholds_gamma)
    converted = gradient_writer.match(w, h)
    if args.invert_color:
        converted = invert_image(converted)
    cv2.imwrite(args.save_path, converted)

    elapsed = time.perf_counter() - start
    print(f"Completed: spent {elapsed:.6f} seconds")

if __name__ == '__main__':
    main()
