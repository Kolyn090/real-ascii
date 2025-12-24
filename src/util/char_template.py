import numpy as np
from PIL.ImageFont import FreeTypeFont

class CharTemplate:
    def __init__(self,
                 char: str,
                 image_font: FreeTypeFont,
                 char_bound: tuple[int, int],
                 img: np.ndarray,
                 img_binary: np.ndarray,
                 img_small: np.ndarray,
                 img_projection: np.ndarray):
        self.char = char
        self.image_font = image_font
        self.char_bound = char_bound
        self.img = img
        self.img_binary = img_binary
        self.img_small = img_small
        self.img_projection = img_projection

    def __eq__(self, other):
        return isinstance(other, CharTemplate) and \
            (self.char, self.image_font.path, self.char_bound) == \
            (other.char, other.image_font.path, other.char_bound)

    def __hash__(self):
        return hash((self.char, self.image_font.path, self.char_bound))

    def __str__(self):
        return f"{{'{self.char}'{self.char_bound}}}"

class PositionalCharTemplate:
    def __init__(self,
                 char_template: CharTemplate,
                 top_left: tuple[int, int]):
        self.char_template = char_template
        self.top_left = top_left

    def __str__(self):
        return f"{{'{self.char_template.char}'{self.top_left}}}"
