import os
import sys
import cv2
import argparse
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import increase_contrast, invert_image  # type: ignore

def contour(img: np.ndarray, canny1: float, canny2: float,
            gb_size=5, gb_sigmaX=0, kernel_size=2, dilate_iter=1, erode_iter=1,
            contrast_factor=1, contrast_window_size=8, invert_color=False):
    img = increase_contrast(img, contrast_factor, (contrast_window_size, contrast_window_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (gb_size, gb_size), gb_sigmaX)
    edges = cv2.Canny(gray, canny1, canny2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    edges = cv2.erode(edges, kernel, iterations=erode_iter)
    contour_img = cv2.bitwise_not(edges)
    if invert_color:
        contour_img = invert_image(contour_img)
    return contour_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_folder', type=str, default='contour')
    parser.add_argument('--canny1_min', type=int, default=0)
    parser.add_argument('--canny1_max', type=int, default=255)
    parser.add_argument('--canny1_step', type=int, default=5)
    parser.add_argument('--canny2_min', type=int, default=0)
    parser.add_argument('--canny2_max', type=int, default=255)
    parser.add_argument('--canny2_step', type=int, default=5)
    parser.add_argument('--gb_size', type=int, default=5)
    parser.add_argument('--gb_sigmaX', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--dilate_iter', type=int, default=1)
    parser.add_argument('--erode_iter', type=int, default=1)
    parser.add_argument('--contrast_factor', type=float, default=1)
    parser.add_argument('--contrast_window_size', type=int, default=8)
    parser.add_argument('--invert_color', action='store_true')
    args = parser.parse_args()
    shutil.rmtree(args.save_folder)
    os.makedirs(args.save_folder, exist_ok=True)

    img = cv2.imread(args.image_path)
    canny1s = list(range(args.canny1_min, args.canny1_max, args.canny1_step))
    canny2s = list(range(args.canny2_min, args.canny2_max, args.canny2_step))
    for canny1 in canny1s:
        for canny2 in canny2s:
            c = contour(img, canny1, canny2,
                        gb_size=args.gb_size,
                        gb_sigmaX=args.gb_sigmaX,
                        kernel_size=args.kernel_size,
                        dilate_iter=args.dilate_iter,
                        erode_iter=args.erode_iter,
                        contrast_factor=args.contrast_factor,
                        contrast_window_size=args.contrast_window_size,
                        invert_color=args.invert_color)
            save_path = os.path.join(args.save_folder,
                                     f"contour_{canny1}_{canny2}.png",)
            cv2.imwrite(save_path, c)

if __name__ == '__main__':
    main()
