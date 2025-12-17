import os.path
import sys
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str, default='ascii_art.png')
    parser.add_argument('--resize_method', type=str, default='nearest_neighbor')
    parser.add_argument('--resize_factor', type=float, default=1)
    parser.add_argument('--contrast_factor', type=float, default=1)
    parser.add_argument('--sigma_s', type=int, default=5)
    parser.add_argument('--sigma_r', type=float, default=0.5)
    parser.add_argument('--thresholds_gamma', type=float, default=0.5)
    parser.add_argument('--palette_path', type=str, default='../../resource/gradient_char_files/palette.txt')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--char_bound_width', type=int, default=13)
    parser.add_argument('--char_bound_height', type=int, default=22)


    args = parser.parse_args()
    save_folder = args.save_path

    save_to_folder = True
    templates = ShadeArgUtil.get_palette(args.palette_path)
    img = cv2.imread(args.image_path)
    img = increase_contrast(img, args.contrast_factor)
    img = resize_bilinear(img, args.resize_factor)
    img = smooth_colors(img, sigma_s=args.sigma_s, sigma_r=args.sigma_r)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    gradient_writer = GradientWriter()
    gradient_writer.save_to_folder = save_to_folder
    gradient_writer.save_folder = save_folder
    gradient_writer.templates = templates
    gradient_writer.assign_gradient_imgs(img, args.thresholds_gamma)
    converted = gradient_writer.match(w, h)
    cv2.imwrite(args.save_path, converted)

if __name__ == '__main__':
    main()
