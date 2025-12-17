import os
import sys
import cv2
import math
import argparse
from PIL import ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from arg_util import TraceArgUtil  # type: ignore
from palette_template import PaletteTemplate  # type: ignore
from static import resize_nearest_neighbor, resize_bilinear, invert_image  # type: ignore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str, default='ascii_art.png')
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--chars', type=str, default='ascii')
    parser.add_argument('--font', type=str, default='C:/Windows/Fonts/consolab.ttf')
    parser.add_argument('--font_size', type=int, default=24)
    parser.add_argument('--char_bound_width', type=int, default=13)
    parser.add_argument('--char_bound_height', type=int, default=22)
    parser.add_argument('--resize_method', type=str, default='nearest_neighbor')
    parser.add_argument('--invert_color', action='store_true')
    parser.add_argument('--chars_file_path', type=str, default='../../resource/char_files/chars_file.txt')

    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--matching_method', type=str, default='fast')
    parser.add_argument('--vector_ratio', type=float, default=0.5)
    parser.add_argument('--vector_top_k', type=int, default=5)

    args = parser.parse_args()

    chars = TraceArgUtil.get_chars(args.chars, args.chars_file_path)
    template = PaletteTemplate(
        layer=0,
        chars=chars,
        imageFont=ImageFont.truetype(args.font, args.font_size),
        char_bound=(args.char_bound_width, args.char_bound_height),
        approx_ratio=args.vector_ratio,
        vector_top_k=args.vector_top_k,
        match_method=args.matching_method
    )

    img = cv2.imread(args.image_path)
    img = TraceArgUtil.resize(args.resize_method, img, args.factor)
    h, w = img.shape[:2]

    slicer = Slicer(args.max_workers)
    cells = slicer.slice(img, (args.char_bound_width, args.char_bound_height))
    writer = template.create_writer(args.max_workers)
    converted = writer.match_cells(cells, w, h)[0]
    converted = converted[0:math.floor(h / args.char_bound_height) * args.char_bound_height,
                            0:math.floor(w / args.char_bound_width) * args.char_bound_width]
    if args.invert_color:
        converted = invert_image(converted)
    cv2.imwrite(args.save_path, converted)

if __name__ == '__main__':
    main()
