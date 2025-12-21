import json
import os.path

import numpy as np
from PIL import ImageFont

from palette_template import PaletteTemplate
from static import resize_nearest_neighbor, resize_bilinear, invert_image
from color_util import (process_image_blocks, blend_ascii_with_color,
                        copy_black_pixels, average_color_block,
                        PositionalColor, reassign_positional_colors)

class TraceArgUtil:
    @staticmethod
    def get_chars(code: str, file_path='../../resource/char_files/chars_file.txt') -> list[str]:
        match code:
            case 'ascii':
                return TraceArgUtil._get_all_ascii()
            case 'file':
                if not os.path.exists(file_path):
                    return []
                return TraceArgUtil._get_from_file(file_path)
        return []

    @staticmethod
    def _get_all_ascii() -> list[str]:
        return [chr(i) for i in range(128)]

    @staticmethod
    def _get_from_file(file_path: str) -> list[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return list(dict.fromkeys(c for c in f.read() if c != '\n'))

    @staticmethod
    def resize(code: str, img: np.ndarray, factor: int) -> np.ndarray:
        match code:
            case 'nearest_neighbor':
                return resize_nearest_neighbor(img, factor)
            case 'bilinear':
                return resize_bilinear(img, factor)
        return resize_nearest_neighbor(img, factor)

class ShadeArgUtil:
    @staticmethod
    def get_palette_json(file_path: str) -> list[PaletteTemplate]:
        def parse_template(obj: dict) -> PaletteTemplate:
            return PaletteTemplate(
                obj["layer"],
                list(dict.fromkeys(c for c in obj["chars"] if c != '\n')),
                ImageFont.truetype(obj["font"], int(obj["font_size"])),
                (obj["char_bound_width"], obj["char_bound_height"]),
                obj["approx_ratio"],
                obj["vector_top_k"],
                obj["match_method"]
            )

        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            name = content["name"]
            templates = content["templates"]
            print(f"Reading palette from {name}.")
            for template in templates:
                result.append(parse_template(template))
        return result

class ColorArgUtil:
    @staticmethod
    def color_image(option: str,
                    ascii_img: np.ndarray,
                    original_img: np.ndarray,
                    cell_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[PositionalColor]] | None:
        if ascii_img is None or original_img is None:
            return None

        match option:
            case 'original':
                return ColorArgUtil.color_original(ascii_img,
                                                   original_img,
                                                   cell_size)
        return None

    @staticmethod
    def color_original(ascii_img: np.ndarray,
                       original_img: np.ndarray,
                       cell_size: tuple[int, int]) \
            -> tuple[np.ndarray, np.ndarray, list[PositionalColor]]:
        color_blocks, p_cs = process_image_blocks(original_img, cell_size, average_color_block)
        color_converted = blend_ascii_with_color(ascii_img, color_blocks, 1)
        color_converted = copy_black_pixels(ascii_img, color_converted)
        return color_converted, color_blocks, p_cs

def test():
    templates = ShadeArgUtil.get_palette_json('../../resource/palette_files/palette_default.json')
    for template in templates:
        print(template)

if __name__ == '__main__':
    test()
