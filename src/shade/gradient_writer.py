import os
import sys
import math

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gradient_divide import divide

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import (resize_nearest_neighbor, resize_bilinear, invert_image,  # type: ignore
                    floor_fill, increase_contrast, to_grayscale, smooth_colors)  # type: ignore
from char_template import CharTemplate, PositionalCharTemplate  # type: ignore
from slicer import Cell, Slicer  # type: ignore
from palette_template import PaletteTemplate  # type: ignore

class GradientWriter:
    def __init__(self,
                 templates: list[PaletteTemplate],
                 max_workers: int):
        self.templates = templates
        self.max_workers = max_workers
        self.gradient_imgs: list[np.ndarray] = []
        self.char_rank: dict[str, int] = dict()

    def assign_gradient_imgs(self, img_gray: np.ndarray, thresholds_gamma: float):
        self.gradient_imgs = divide(img_gray, len(self.templates), thresholds_gamma)
        # count = 0
        # for gradient_img in self.gradient_imgs:
        #     cv2.imwrite(f"test_writer/gradient_{count}.png", gradient_img)
        #     count += 1

    def match(self, w: int, h: int) -> tuple[np.ndarray, list[PositionalCharTemplate]]:
        p_ct_lists: list[list[PositionalCharTemplate]] = []
        for i in range(len(self.templates)):
            template = self.templates[i]
            writer = template.create_writer(self.max_workers)
            img = self.gradient_imgs[i]
            img = invert_image(img)
            slicer = Slicer()
            cells = slicer.slice(img, self.templates[i].char_bound)
            h, w = img.shape[:2]
            _, p_cts = writer.match_cells(cells, w, h)
            p_ct_lists.append(p_cts)

            # Test only
            # med_img = self.stack_to_img(p_cts, w, h)
            # cv2.imwrite(f"test_writer/med_{i}.png", med_img)

        stacks = self.stack(p_ct_lists)
        # result_img = np.zeros((h, w, 3), dtype=np.uint8)
        #
        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     list(executor.map(lambda cell: self._paste_to_img(cell, result_img), stacks))
        #
        # result_img = invert_image(result_img)
        # large_char_bound = self.get_large_char_bound()
        # result_img = result_img[0:math.floor(h / large_char_bound[1]) * large_char_bound[1],
        #                         0:math.floor(w / large_char_bound[0]) * large_char_bound[0]]
        # return result_img, stacks
        return self.stack_to_img(stacks, w, h), stacks

    def stack_to_img(self, p_cts: list[PositionalCharTemplate], w: int, h: int) \
            -> np.ndarray:
        stacks = p_cts
        result_img = np.zeros((h, w, 3), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(lambda cell: self._paste_to_img(cell, result_img), stacks))

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
    def _paste_to_img(p_ct: PositionalCharTemplate, result_img: np.ndarray):
        template = p_ct.char_template.img
        top_left = p_ct.top_left
        bottom_right_y = top_left[1] + template.shape[0]
        bottom_right_x = top_left[0] + template.shape[1]
        result_img[top_left[1]:bottom_right_y, top_left[0]:bottom_right_x] = template

    def stack(self, p_ct_lists: list[list[PositionalCharTemplate]]) -> list[PositionalCharTemplate]:
        self._assign_template_rank()
        table: dict[tuple[int, int], CharTemplate] = dict()
        for p_ct_list in reversed(p_ct_lists):
        # for p_ct_list in p_ct_lists:
            for p_ct in p_ct_list:
                char_template = p_ct.char_template
                top_left = p_ct.top_left
                self._add_to_table(table, top_left, char_template)

        result = []
        for top_left, char_template in table.items():
            p_ct = PositionalCharTemplate(char_template, top_left)
            result.append(p_ct)
        return result

    def _add_to_table(self, table: dict[tuple[int, int], CharTemplate],
                      top_left: tuple[int, int],
                      char_template: CharTemplate):
        if top_left in table:
            # Prioritize the character in the lower rank (0 is the highest rank)
            if self._compare_template_char(char_template.char, table[top_left].char):
                table[top_left] = char_template
        else:
            table[top_left] = char_template

    def _compare_template_char(self, tc1: str, tc2: str) -> bool:
        """
        Compare two template chars.

        :param tc1: template char 1
        :param tc2: template char 2
        :return: Return True if tc1 has lower rank in templates.
        Otherwise, False.
        """
        return self.char_rank[tc1] > self.char_rank[tc2]

    def _assign_template_rank(self):
        count = 0
        for template in self.templates:
            for char in template.chars:
                self.char_rank[char] = count
                count += 1
        # Force space to have the guaranteed highest rank
        self.char_rank[" "] = -1
