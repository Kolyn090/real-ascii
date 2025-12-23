import os
import sys
import cv2
import numpy as np

from eg_divide import divide
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shade')))
from gradient_writer import GradientWriter  # type: ignore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from palette_template import PaletteTemplate  # type: ignore
from static import increase_contrast, resize_bilinear, smooth_colors, to_grayscale  # type: ignore
from arg_util import ShadeArgUtil  # type: ignore
from char_template import PositionalCharTemplate  # type: ignore

class EdgeGradientWriter:
    def __init__(self,
                 templates: list[PaletteTemplate],
                 max_workers: int):
        self.gradient_writer = GradientWriter(templates, max_workers)

    def assign_gradient_imgs(self, img_gray: np.ndarray,
                             sigmaX: float,
                             thresholds_gamma: float,
                             ksize: int,
                             gx: int,
                             gy: int):
        self.gradient_writer.gradient_imgs = divide(img_gray,
                                                    len(self.gradient_writer.templates),
                                                    sigmaX=sigmaX,
                                                    gamma=thresholds_gamma,
                                                    ksize=ksize,
                                                    gx=gx,
                                                    gy=gy)

    def match(self, w: int, h: int) -> tuple[np.ndarray, list[PositionalCharTemplate]]:
        return self.gradient_writer.match(w, h)

def test():
    factor = 4
    thresholds_gamma = 1.8
    sigmaX = 0.5
    ksize = 9
    gx = 3
    gy = 3
    img_path = '../../resource/f_input/ultraman-nexus.png'
    save_folder = 'test_writer'
    save_to_folder = True
    img = cv2.imread(img_path)
    img = increase_contrast(img, 2)

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    # if save_to_folder:
    #     cv2.imwrite(os.path.join(save_folder, "original_img.png"), img)

    img = resize_bilinear(img, factor)
    img = smooth_colors(img, sigma_s=10, sigma_r=1.5)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    # if save_to_folder:
    #     cv2.imwrite(os.path.join(save_folder, "img.png"), img)

    templates = ShadeArgUtil.get_palette_json('../../resource/palette_files/palette_default.json')
    for template in templates:
        template.match_method = 'slow'

    eg_writer = EdgeGradientWriter(templates, max_workers=16)
    eg_writer.assign_gradient_imgs(img, sigmaX, thresholds_gamma, ksize, gx, gy)

    for i in range(len(eg_writer.gradient_writer.gradient_imgs)):
        if save_to_folder:
            cv2.imwrite(os.path.join(save_folder, f"gradient_{i}.png"), eg_writer.gradient_writer.gradient_imgs[i])

    converted = eg_writer.match(w, h)
    if save_to_folder:
        cv2.imwrite(os.path.join(save_folder, "test.png"), converted)

if __name__ == '__main__':
    test()
