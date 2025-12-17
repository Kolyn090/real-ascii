import cv2
import math
import numpy as np
from typing import Callable
from PIL.ImageFont import FreeTypeFont
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

from slicer import Cell
from static import to_binary_strong

class CharTemplate:
    def __init__(self):
        self.char: str | None = None
        self.template: np.ndarray | None = None
        self.template_binary: np.ndarray | None = None
        self.template_small: np.ndarray | None = None
        self.projection: np.ndarray | None = None

class PositionalCharTemplate:
    def __init__(self):
        self.char_template: CharTemplate | None = None
        self.top_left: tuple[int, int] | None = None

class Writer:
    def __init__(self,
                 imageFont: FreeTypeFont,
                 max_workers: int,
                 char_bound: tuple[int, int],
                 approx_ratio: float,
                 match_method: str,
                 vector_top_k: int):
        self.imageFont = imageFont
        self.max_workers = max_workers
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio
        self.get_most_similar = self.get_matching_method(match_method)
        self.vector_top_k = vector_top_k

        self.char_templates: list[CharTemplate] = []
        self.space_template = CharTemplate()
        self.approx_size = (7, 12)

    def get_matching_method(self, method: str) -> Callable[[np.ndarray], CharTemplate]:
        match method:
            case 'slow':
                return self._get_most_similar_slow
            case 'optimized':
                return self._get_most_similar
            case 'fast':
                return self._get_most_similar_fast
            case 'vector':
                return self._get_most_similar_vector
        return self._get_most_similar_fast

    def match_cells(self, cells: list[Cell],
                    w: int, h: int) -> tuple[np.ndarray, list[PositionalCharTemplate]]:
        result_img = np.zeros((h, w, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            templates = list(executor.map(lambda cell: self._paste_to_img(cell, result_img), cells))
        return result_img, templates

    def _paste_to_img(self, cell: Cell, result_img: np.ndarray) -> PositionalCharTemplate:
        """
        Paste the cell to the final image.
        Return the most similar template to the given cell.

        :param cell: The cell
        :param result_img: The final image
        :return: The most similar template to cell
        """
        most_similar = self.get_most_similar(cell.img)
        template = most_similar.template
        top_left = cell.top_left
        bottom_right_y = top_left[1] + template.shape[0]
        bottom_right_x = top_left[0] + template.shape[1]
        result_img[top_left[1]:bottom_right_y, top_left[0]:bottom_right_x] = template

        result = PositionalCharTemplate()
        result.char_template = most_similar
        result.top_left = top_left
        return result

    def _get_most_similar_slow(self, img: np.ndarray) -> CharTemplate:
        """
        Get the most similar template to the given cell.
        Warning: if the image is empty, the result template
        is guaranteed to be 'space'.

        :param img: The cell img
        :return: The most similar template to cell
        """
        if len(self.char_templates) == 0:
            raise Exception("You have not assigned any template yet.")

        # All white → space
        if np.all(img):
            return self.space_template

        best_score = -1
        best_template = None

        h, w = img.shape[:2]

        for char_template in self.char_templates:
            template = char_template.template
            template_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_NEAREST)
            img_gray = self._ensure_gray(img)
            template_gray = self._ensure_gray(template_resized)
            score = np.sum(img_gray == template_gray) / (w * h)
            if score > best_score:
                best_score = score
                best_template = char_template
        return best_template

    def _get_most_similar(self, img: np.ndarray) -> CharTemplate:
        """
        Get the most similar template to the given cell.
        Warning: if the image is empty, the result template
        is guaranteed to be 'space'.

        :param img: The cell img
        :return: The most similar template to cell
        """
        if len(self.char_templates) == 0:
            raise Exception("You have not assigned any template yet.")

        img_bin = (img > 0)
        # All white → space
        if np.all(img_bin):
            return self.space_template

        best_score = -1
        best_template = None

        h, w = img_bin.shape[:2]

        for char_template in self.char_templates:
            template = char_template.template_binary
            template_resized = cv2.resize(
                template, (w, h), interpolation=cv2.INTER_NEAREST
            )

            template_bin = (template_resized > 0)

            # Hamming similarity
            same = np.count_nonzero(img_bin == template_bin)
            score = same / (w * h)

            if score > best_score:
                best_score = score
                best_template = char_template

        return best_template

    def _get_most_similar_fast(self, img: np.ndarray):
        if len(self.char_templates) == 0:
            raise Exception("You have not assigned any template yet.")

        img_bin = (img > 0)

        # All white → space
        if np.all(img_bin):
            return self.space_template

        best_score = -1
        best_template = None

        for ct in self.char_templates:
            template_bin = (ct.template_binary > 0)

            # Boolean comparison + count
            score = np.count_nonzero(img_bin == template_bin) / img_bin.size

            if score > best_score:
                best_score = score
                best_template = ct

            # Early exit if perfect match
            if best_score == 1.0:
                break

        return best_template

    def _get_most_similar_vector(self, img: np.ndarray):
        img_bin = (img > 0).astype(np.uint8)
        if np.all(img_bin):
            return self.space_template

        # --- Stage 1: fast approximate match ---
        img_small = cv2.resize(img_bin, self.approx_size, interpolation=cv2.INTER_NEAREST)
        img_feat = img_small.ravel()

        templates_stack = np.stack([ct.projection for ct in self.char_templates])

        # L1 / Hamming distance (more robust than equality)
        dists = np.sum(np.abs(templates_stack - img_feat), axis=1)

        # Select top-K best candidates
        top_idx = np.argpartition(dists, self.vector_top_k)[:self.vector_top_k]

        # --- Stage 2: accurate full-resolution recheck ---
        best_score = -1
        best_template = None

        for idx in top_idx:
            ct = self.char_templates[idx]
            template_bin = (ct.template_binary > 0)

            score = np.count_nonzero(img_bin == template_bin) / img_bin.size

            if score > best_score:
                best_score = score
                best_template = ct

        return best_template

    @staticmethod
    def _ensure_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def assign_char_templates(self, chars: list[str]) -> list[CharTemplate]:
        result = []
        for char in chars:
            char_template = self._create_char_template(char, self.imageFont)
            result.append(char_template)
        self.char_templates = result
        self.space_template = self._create_char_template(" ", self.imageFont)
        return result

    def _create_char_template(self, char: str, imageFont: ImageFont) -> CharTemplate:
        self.approx_size = (math.floor(self.char_bound[0] * self.approx_ratio),
                            math.floor(self.char_bound[1] * self.approx_ratio))
        img = Image.new("RGB", self.char_bound, "white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=imageFont, fill="black")
        char_template = CharTemplate()
        char_template.char = char
        char_template.template = np.array(img)
        char_template.template_binary = to_binary_strong(char_template.template)
        template_small = cv2.resize(char_template.template_binary, self.approx_size, interpolation=cv2.INTER_NEAREST)
        template_small = to_binary_strong(template_small)
        char_template.template_small = template_small
        char_template.projection = template_small.ravel()
        return char_template
