import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

from slicer import Cell, Slicer

class CharTemplate:
    def __init__(self):
        self.char: str | None = None
        self.template: np.ndarray | None = None

class PositionalCharTemplate:
    def __init__(self):
        self.char_template: CharTemplate | None = None
        self.top_left: tuple[int, int] | None = None

class Writer:
    def __init__(self):
        self.font_size = 24
        self.font = 'C:/Windows/Fonts/consolab.ttf'
        self.char_templates: list[CharTemplate] = []
        self.space_template = CharTemplate()

    def match_cells(self, cells: list[Cell],
                    w: int, h: int) -> tuple[np.ndarray, list[PositionalCharTemplate]]:
        result_img = np.zeros((h, w, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(executor.map(lambda cell: self.paste_to_img(cell, result_img), cells))
        return result_img, templates

    def paste_to_img(self, cell: Cell, result_img: np.ndarray) -> PositionalCharTemplate:
        """
        Paste the cell to the final image.
        Return the most similar template to the given cell.

        :param cell: The cell
        :param result_img: The final image
        :return: The most similar template to cell
        """
        most_similar = self.get_most_similar(cell)
        template = most_similar.template
        top_left = cell.top_left
        bottom_right_y = top_left[1] + template.shape[0]
        bottom_right_x = top_left[0] + template.shape[1]
        result_img[top_left[1]:bottom_right_y, top_left[0]:bottom_right_x] = template

        result = PositionalCharTemplate()
        result.char_template = most_similar
        result.top_left = top_left
        return result

    def get_most_similar(self, cell: Cell) -> CharTemplate:
        """
        Get the most similar template to the given cell.
        Warning: if the image is empty, the result template
        is guaranteed to be 'space'.

        :param cell: The cell
        :return: The most similar template to cell
        """
        if len(self.char_templates) == 0:
            raise Exception("You have not assigned any template yet.")

        best_score = -1
        best_template = None

        img = cell.img
        h, w = img.shape[:2]

        if np.all(img == 255):
            return self.space_template

        for char_template in self.char_templates:
            template = char_template.template
            template_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_NEAREST)
            img_gray = self.ensure_gray(img)
            template_gray = self.ensure_gray(template_resized)
            score = np.sum(img_gray == template_gray) / (w * h)
            if score > best_score:
                best_score = score
                best_template = char_template
        return best_template

    @staticmethod
    def ensure_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def assign_char_templates(self, chars: list[str]) -> list[CharTemplate]:
        imageFont = ImageFont.truetype(self.font, self.font_size)
        result = []
        for char in chars:
            char_template = self.create_char_template(char, imageFont)
            result.append(char_template)
        self.char_templates = result
        self.space_template = self.create_char_template(" ", imageFont)
        return result

    @staticmethod
    def create_char_template(char: str, imageFont: ImageFont) -> CharTemplate:
        img = Image.new("RGB", (13, 22), "white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=imageFont, fill="black")
        char_template = CharTemplate()
        char_template.char = char
        char_template.template = np.array(img)
        return char_template

def test_char_templates():
    save_to_folder = False
    save_folder = 'chars'
    chars = [chr(i) for i in range(128)]
    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    writer = Writer()
    char_templates = writer.assign_char_templates(chars)
    for char_template in char_templates:
        char = char_template.char
        template = char_template.template
        print(char)
        save_path = os.path.join(save_folder, f"char_{ord(char)}.png")
        if save_to_folder:
            cv2.imwrite(save_path, template)

def test_match_cells():
    img_path = '../trace/test/contour_10_100.png'
    save_folder = 'test'
    save_to_folder = True
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    chars = [chr(i) for i in range(128)]

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    slicer = Slicer()
    cells = slicer.slice(img, (13, 22))
    writer = Writer()
    writer.assign_char_templates(chars)
    converted = writer.match_cells(cells, w, h)[0]
    cv2.imwrite(os.path.join(save_folder, 'converted.png'), converted)

if __name__ == '__main__':
    test_match_cells()
