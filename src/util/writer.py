import cv2
import math
import numpy as np
from typing import Callable
from PIL.ImageFont import FreeTypeFont
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

from slicer import Cell
from static import to_binary_strong
from char_template import CharTemplate, PositionalCharTemplate

class Writer:
    def __init__(self,
                 image_font: FreeTypeFont,
                 max_workers: int,
                 char_bound: tuple[int, int],
                 approx_ratio: float,
                 match_method: str,
                 vector_top_k: int,
                 chars: list[str],
                 smoothing: bool,
                 override_widths: dict[str, int] | None = None,
                 override_weights: dict[tuple[str, int], float] | None = None):
        self.image_font = image_font
        self.max_workers = max_workers
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio if approx_ratio > 0 else 0.5
        self.get_most_similar = self.get_matching_method(match_method)
        self.vector_top_k = vector_top_k if vector_top_k > 0 else 5

        self.char_templates: list[CharTemplate] = []
        self.space_template = None
        self.approx_size = (7, 12)
        self.override_widths = override_widths
        self.override_weights = override_weights
        self.smoothing = smoothing

        self._assign_char_templates(chars)

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
            char_templates = list(executor.map(lambda cell: self._paste_to_img(cell, result_img), cells))
        return result_img, char_templates

    def _paste_to_img(self, cell: Cell, result_img: np.ndarray) -> PositionalCharTemplate:
        """
        Paste the cell to the final image.
        Return the most similar template to the given cell.

        :param cell: The cell
        :param result_img: The final image
        :return: The most similar template to cell
        """
        most_similar = self.get_most_similar(cell.img)
        if self.smoothing:
            template = most_similar.img
        else:
            template = most_similar.img_binary
        top_left = cell.top_left
        # bottom_right_y = top_left[1] + template.shape[0]
        # bottom_right_x = top_left[0] + template.shape[1]

        # template: (H, W) or (H, W, 3)
        h, w = template.shape[:2]

        # ensure template has 3 channels
        if template.ndim == 2:  # grayscale
            template_to_paste = np.stack([template] * 3, axis=-1)
        elif template.ndim == 3 and template.shape[2] == 3:  # already RGB
            template_to_paste = template
        else:
            raise ValueError(f"Unsupported template shape: {template.shape}")

        # paste into result_img
        result_img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = template_to_paste

        return PositionalCharTemplate(most_similar, top_left)

    def _get_most_similar_slow(self, img: np.ndarray) -> CharTemplate:
        """
        Get the most similar template to the given cell.
        Warning: if the image is empty, the result template
        is guaranteed to be 'space'.

        :param img: The cell img
        :return: The most similar template to cell
        """

        def ensure_gray(image):
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image

        if len(self.char_templates) == 0:
            raise Exception("You have not assigned any template yet.")

        # All white → space
        if np.all(img):
            return self.space_template

        best_score = -1
        best_template = None

        h, w = img.shape[:2]

        for char_template in self.char_templates:
            template = char_template.img
            template_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_NEAREST)
            img_gray = ensure_gray(img)
            template_gray = ensure_gray(template_resized)
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

        img_bin = to_binary_strong(img)
        # All white → space
        if np.all(img_bin):
            return self.space_template

        best_score = -1
        best_template = None

        h, w = img_bin.shape[:2]

        for char_template in self.char_templates:
            template = char_template.img_binary
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

        img_bin = to_binary_strong(img)

        # All white → space
        if np.all(img_bin):
            return self.space_template

        best_score = -1
        best_template = None

        for ct in self.char_templates:
            template_bin = (ct.img_binary > 0)

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
        img_bin = to_binary_strong(img)
        if np.all(img_bin):
            return self.space_template

        # --- Stage 1: fast approximate match ---
        img_small = cv2.resize(img_bin, self.approx_size, interpolation=cv2.INTER_NEAREST)
        img_feat = img_small.ravel()

        templates_stack = np.stack([ct.img_projection for ct in self.char_templates])

        # L1 / Hamming distance (more robust than equality)
        dists = np.sum(np.abs(templates_stack - img_feat), axis=1)

        # Select top-K best candidates
        top_idx = np.argpartition(dists, self.vector_top_k)[:self.vector_top_k]

        # --- Stage 2: accurate full-resolution recheck ---
        best_score = -1
        best_template = None

        for idx in top_idx:
            ct = self.char_templates[idx]
            template_bin = (ct.img_binary > 0)

            score = np.count_nonzero(img_bin == template_bin) / img_bin.size

            if score > best_score:
                best_score = score
                best_template = ct

        return best_template

    def _assign_char_templates(self, chars: list[str]):
        result = []
        for char in chars:
            char_template = self._create_char_template(char)
            result.append(char_template)
        self.char_templates = result
        self.space_template = self._create_char_template(" ")

    def _create_char_template(self, char: str) -> CharTemplate:
        char_bound = self._get_char_bound(char)
        self.approx_size = (math.floor(char_bound[0] * self.approx_ratio),
                            math.floor(char_bound[1] * self.approx_ratio))
        img = Image.new("RGB", char_bound, "white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=self.image_font, fill="black")

        template = np.array(img)
        template_binary = to_binary_strong(template)
        template_small = cv2.resize(template_binary, self.approx_size, interpolation=cv2.INTER_NEAREST)
        template_small = to_binary_strong(template_small)
        char_template = CharTemplate(
            char=char,
            image_font=self.image_font,
            char_bound=char_bound,
            img=template,
            img_binary=template_binary,
            img_small=template_small,
            img_projection=template_small.ravel()
        )
        return char_template

    def _get_char_bound(self, char: str) -> tuple[int, int]:
        if self.override_widths is not None and char in self.override_widths:
            return self.override_widths[char], self.char_bound[1]
        return self.char_bound
