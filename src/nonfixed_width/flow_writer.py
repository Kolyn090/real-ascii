import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PIL.ImageFont import FreeTypeFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from char_template import CharTemplate, PositionalCharTemplate  # type: ignore
from static import to_binary_middle, to_binary_strong  # type: ignore
from image_padding import pil_pad_columns  # type: ignore

class FlowWriter:
    """

    :param binary_threshold: if using fast match method, the long img needs to be converted to binary first.
    ! Important: the match method used in FlowWriter is not the same as
    the one used in Writer. Here only two possible methods are supported:
    1. slow (match by grayscale)
    2. fast (match by binary xor)
    """
    def __init__(self,
                 chars: list[str],
                 char_bound: tuple[int, int],
                 override_widths: dict[str, int] | None,
                 image_font: FreeTypeFont,
                 pad: tuple[int, int],
                 flow_match_method: str,
                 binary_threshold=90,
                 override_weights: dict[tuple[str, int], float] | None = None,
                 maximum_char_width=60):
        self.char_bound = char_bound
        self.override_widths = override_widths
        self.override_weights = override_weights
        self.image_font = image_font
        self.flow_match_method = flow_match_method
        self.pad = pad
        self.binary_threshold = binary_threshold
        self.char_templates = [self._create_char_template(char) for char in chars]
        self.maximum_char_width = maximum_char_width

    def match(self, img: np.ndarray) -> tuple[np.ndarray, list[PositionalCharTemplate]]:
        """

        :param img: A grayscale image
        :return: A tuple of items.
        1. the resulting image
        2. the positional ct that assembled the image
        """
        long_imgs = self._split_into_rows(img, self.char_bound[1])
        rows = []
        seqs = []
        row_num = 0
        for long_img in long_imgs:
            reconstructed = self._tile_row(long_img, row_num)
            seq, score = reconstructed
            seqs.extend(seq)
            seq = [ct.char_template.img_binary for ct in seq]
            rows.append(self.concat_images_left_to_right(seq))
            row_num += 1
        final_img = self.concat_images_top_to_bottom(rows, pad_color=(255, 255, 255))
        return final_img, seqs

    @staticmethod
    def _split_into_rows(img: np.ndarray, row_height: int) -> list[np.ndarray]:
        H = img.shape[0]
        rows = []
        for y in range(0, H - row_height + 1, row_height):
            rows.append(img[y:y + row_height])
        return rows

    @staticmethod
    def concat_images_left_to_right(images: list[np.ndarray]) -> np.ndarray:
        """
        Concatenates a list of images (all same height) horizontally (left to right).
        """
        # check that all images have the same height
        heights = [img.shape[0] for img in images]
        if len(set(heights)) != 1:
            raise ValueError("All images must have the same height")

        # concatenate along the width axis
        concatenated = np.hstack(images)
        return concatenated

    @staticmethod
    def concat_images_top_to_bottom(images: list[np.ndarray], pad_color=(255,255,255)) -> np.ndarray:
        """
        Concatenates images vertically. If widths differ, pad them to the max width.

        pad_color: tuple or scalar
            Color used for padding (RGB for color images, single int for grayscale)
        """
        if not images:
            raise ValueError("Image list is empty")

        # Determine max width and height per image
        widths = [img.shape[1] for img in images]
        max_width = max(widths)

        padded_images = []
        for img in images:
            h, w = img.shape[:2]

            if w == max_width:
                padded_images.append(img)
                continue

            # Create padded canvas
            if img.ndim == 3:  # color image
                pad_img = np.full((h, max_width, img.shape[2]), pad_color, dtype=img.dtype)
            else:  # grayscale
                gray_color = pad_color if not isinstance(pad_color, tuple) else pad_color[0]
                pad_img = np.full((h, max_width),
                                  gray_color, dtype=img.dtype)

            # place image on left, pad on right
            pad_img[:, :w] = img
            padded_images.append(pad_img)

        return np.vstack(padded_images)

    def _tile_row(self, long_img: np.ndarray, row_num: int) \
            -> tuple[list[PositionalCharTemplate], float] | None:
        """

        :param long_img: a gray scale image that has very long width, must have the same height to all templates.
        :return: the assembling char templates, along with the final matching score
        """
        # key = index
        # value = (char template, short image)
        ct_table: dict[int, tuple[CharTemplate, np.ndarray]] = dict()
        if self.flow_match_method == 'fast':
            long_img = to_binary_middle(long_img, self.binary_threshold)
            long_img = long_img.astype(bool)
            for i in range(len(self.char_templates)):
                short_img = self.char_templates[i].img_binary.astype(bool)
                ct_table[i] = (self.char_templates[i], short_img)
        else:
            for i in range(len(self.char_templates)):
                short_img = self.char_templates[i].img  # grayscale
                ct_table[i] = (self.char_templates[i], short_img)

        H, W = long_img.shape

        # --- DP arrays ---
        dp = [float('inf')] * (W + 1)
        choice: list[CharTemplate | None] = [None] * (W + 1)
        prev: list[int | None] = [None] * (W + 1)
        dp[0] = 0

        # --- DP ---
        for x in range(W + 1):
            if dp[x] == float('inf'):
                continue

            for i in range(len(self.char_templates)):
                tile = ct_table[i][1]
                _, w = tile.shape
                nx = min(x + w, W)
                region = long_img[:, x:nx]
                tile_crop = tile[:, :region.shape[1]]

                c = 0
                if self.flow_match_method == 'fast':
                    c = np.count_nonzero(region != tile_crop)
                else:  # slow
                    c = self._mse(region, tile_crop)

                if dp[x] + c < dp[nx]:
                    dp[nx] = dp[x] + c
                    choice[nx] = ct_table[i][0]
                    prev[nx] = x

        # --- reconstruct solution ---
        seq: list[PositionalCharTemplate] = []

        x = W
        while x > 0:
            seq.insert(0, PositionalCharTemplate(choice[x], (x-choice[x].char_bound[0], row_num * self.char_bound[1])))
            x = prev[x]

        return seq, dp[W]

    @staticmethod
    def _mse(img1: np.ndarray, img2: np.ndarray) -> float:
        # img1 = img1.astype(np.float32)
        # img2 = img2.astype(np.float32)

        # Assume both are grayscale
        mse = np.mean((img1 - img2) ** 2)

        return mse / (255.0 ** 2)

    def _create_char_template(self, char: str) -> CharTemplate:
        char_bound = self._get_char_bound(char)

        if char_bound[0] <= 0:
            # We want to trim the width exactly to the width of the character
            # AKA, no all-white columns should remain
            img = Image.new("RGB", (self.maximum_char_width, char_bound[1]), "white")
        else:
            img = Image.new("RGB", char_bound, "white")

        draw = ImageDraw.Draw(img)

        # Get text bounding box
        bbox = draw.textbbox((0, 0), char, font=self.image_font)

        text_w = bbox[2] - bbox[0]

        img_w, img_h = img.size

        # Center horizontally
        x = (img_w - text_w) // 2

        draw.text((x, 0), char, font=self.image_font, fill="black")

        final_bound = char_bound
        if char_bound[0] <= 0:
            # ---- Trim all-white columns ----
            gray = img.convert("L")
            inv = ImageOps.invert(gray)

            bbox = inv.getbbox()
            if bbox:
                left, top, right, bottom = bbox
                # keep full height, only crop columns
                img = img.crop((left, 0, right, img.height))
            # Pad white columns to image.
            # For example, if the width is -1, pad 1 column to both sides of the image
            img = pil_pad_columns(img, -char_bound[0])
            final_bound = (img.size[0], char_bound[1])

        template = np.array(img)
        template_binary = to_binary_strong(template)
        template_small = template_binary
        template_small = to_binary_strong(template_small)
        char_template = CharTemplate(
            char=char,
            image_font=self.image_font,
            char_bound=final_bound,
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
