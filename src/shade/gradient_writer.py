import math
import os
import cv2
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gradient_divide import divide

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import (resize_nearest_neighbor, resize_bilinear, invert_image,  # type: ignore
                    floor_fill, increase_contrast, to_grayscale, smooth_colors)  # type: ignore
from writer import Writer, CharTemplate, PositionalCharTemplate  # type: ignore
from slicer import Cell, Slicer  # type: ignore
from palette_template import PaletteTemplate  # type: ignore
from arg_util import ShadeArgUtil  # type: ignore

class GradientWriter:
    def __init__(self,
                 templates: list[PaletteTemplate],
                 max_workers):
        self.templates = templates
        self.max_workers = max_workers
        self.gradient_imgs: list[np.ndarray] = []
        self.template_rank: dict[str, int] = dict()

    def assign_gradient_imgs(self, img_gray: np.ndarray, thresholds_gamma: float):
        self.gradient_imgs = divide(img_gray, len(self.templates), thresholds_gamma)

    def match(self, w: int, h: int) -> np.ndarray:
        p_ct_lists: list[list[PositionalCharTemplate]] = []
        for i in range(len(self.templates)):
            writer = Writer()
            writer.assign_get_most_similar(self.templates[i].match_method)
            writer._assign_char_templates(self.templates[i].chars)

            img = self.gradient_imgs[i]
            img = invert_image(img)
            slicer = Slicer()
            cells = slicer.slice(img, self.templates[i].char_bound)
            h, w = img.shape[:2]
            _, p_cts = writer.match_cells(cells, w, h)
            p_ct_lists.append(p_cts)
        stacks = self.stack(p_ct_lists)
        result_img = np.zeros((h, w, 3), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(lambda cell: self.paste_to_img(cell, result_img), stacks))

        result_img = invert_image(result_img)
        large_char_bound = self.get_large_char_bound()
        result_img = result_img[0:math.floor(h / large_char_bound[1]) * large_char_bound[1],
                                0:math.floor(w / large_char_bound[0]) * large_char_bound[0]]
        return result_img

    def get_large_char_bound(self) -> tuple[int, int]:
        result_width = 0
        result_height = 0
        for template in self.templates:
            result_width = max(template.char_bound[0], result_width)
            result_height = max(template.char_bound[1], result_height)
        return result_width, result_height

    @staticmethod
    def paste_to_img(p_ct: PositionalCharTemplate, result_img: np.ndarray):
        template = p_ct.char_template.template
        top_left = p_ct.top_left
        bottom_right_y = top_left[1] + template.shape[0]
        bottom_right_x = top_left[0] + template.shape[1]
        result_img[top_left[1]:bottom_right_y, top_left[0]:bottom_right_x] = template

    def stack(self, p_ct_lists: list[list[PositionalCharTemplate]]) -> list[PositionalCharTemplate]:
        self.assign_template_rank()
        table: dict[tuple[int, int], CharTemplate] = dict()
        for p_ct_list in reversed(p_ct_lists):
        # for p_ct_list in p_ct_lists:
            for p_ct in p_ct_list:
                char_template = p_ct.char_template
                top_left = p_ct.top_left
                self.add_to_table(table, top_left, char_template)

        result = []
        for top_left, char_template in table.items():
            p_ct = PositionalCharTemplate()
            p_ct.top_left = top_left
            p_ct.char_template = char_template
            result.append(p_ct)
        return result

    def add_to_table(self, table: dict[tuple[int, int], CharTemplate],
                     top_left: tuple[int, int],
                     char_template: CharTemplate):
        if top_left in table:
            if self.compare_template_char(char_template.char, table[top_left].char):
                table[top_left] = char_template
        else:
            table[top_left] = char_template

    def compare_template_char(self, tc1: str, tc2: str) -> bool:
        """
        Compare two template chars.

        :param tc1: template char 1
        :param tc2: template char 2
        :return: Return True if tc1 has lower rank in templates.
        Otherwise, False.
        """
        return self.template_rank[tc1] > self.template_rank[tc2]

    def assign_template_rank(self):
        count = 0
        for template in self.templates:
            for char in template.chars:
                self.template_rank[char] = count
                count += 1
        # Force space to have the lowest rank
        self.template_rank[" "] = -1

def test():
    factor = 4
    thresholds_gamma = 0.3
    img_path = '../f_input/prof.jpg'
    save_folder = 'test_writer'
    save_to_folder = True
    img = cv2.imread(img_path)
    img = increase_contrast(img, 2)

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    # if save_to_folder:
    #     cv2.imwrite(os.path.join(save_folder, "original_img.png"), img)

    img = resize_bilinear(img, factor)
    img = smooth_colors(img, sigma_s=1, sigma_r=0.6)
    img = to_grayscale(img)
    h, w = img.shape[:2]

    # if save_to_folder:
    #     cv2.imwrite(os.path.join(save_folder, "img.png"), img)

    templates = ShadeArgUtil.get_palette_json('../../resource/gradient_char_files/palette_default.json')
    gradient_writer = GradientWriter(templates, max_workers=16)
    gradient_writer.assign_gradient_imgs(img, thresholds_gamma)

    for i in range(len(gradient_writer.gradient_imgs)):
        if save_to_folder:
            cv2.imwrite(os.path.join(save_folder, f"gradient_{i}.png"), gradient_writer.gradient_imgs[i])

    gradient_writer.match(w, h)

if __name__ == '__main__':
    test()
