import json
import os.path

import cv2
import numpy as np

from char_template import PositionalCharTemplate
from palette_template import PaletteTemplate
from static import resize_nearest_neighbor, resize_bilinear, invert_image
from color_util import (process_image_blocks, blend_ascii_with_color,
                        copy_black_pixels, blend_pixels, average_color_block,
                        PositionalColor, process_image_blocks_nonfixed_width)

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
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            name = content["name"]
            templates = content["templates"]
            print(f"Reading palette from {name}.")
            for template in templates:
                result.append(PaletteTemplate.read_from_json(template))
        return result

class ColorArgUtil:
    @staticmethod
    def color_image(option: str,
                    ascii_img: np.ndarray,
                    original_img: np.ndarray,
                    cell_size: tuple[int, int],
                    antialiasing=False,
                    invert_ascii=True,
                    are_fixed=False,
                    p_cts: list[PositionalCharTemplate]=None) -> tuple[np.ndarray, np.ndarray, list[PositionalColor]] | None:
        if ascii_img is None or original_img is None:
            return None

        match option:
            case 'original':
                return ColorArgUtil.color_original(ascii_img,
                                                   original_img,
                                                   cell_size,
                                                   antialiasing,
                                                   invert_ascii,
                                                   are_fixed,
                                                   p_cts)
        return None

    @staticmethod
    def color_original(ascii_img: np.ndarray,
                       original_img: np.ndarray,
                       cell_size: tuple[int, int],
                       antialiasing: bool,
                       invert_ascii: bool,
                       are_fixed: bool,
                       p_cts: list[PositionalCharTemplate]) \
            -> tuple[np.ndarray, np.ndarray, list[PositionalColor]]:
        if not are_fixed:
            if p_cts is None:
                raise Exception("Error: using a non-fixed width palette file but positional char template list is not provided.")
            color_blocks, p_cs = process_image_blocks_nonfixed_width(original_img, p_cts, average_color_block)
            h = min(color_blocks.shape[0], ascii_img.shape[0])
            w = min(color_blocks.shape[1], ascii_img.shape[1])
            if invert_ascii:
                ascii_img = invert_image(ascii_img)
            converted = ascii_img[:h, :w]
            color_blocks = color_blocks[:h, :w]
            color_converted = blend_ascii_with_color(converted, color_blocks, 1)
        else:
            color_blocks, p_cs = process_image_blocks(original_img, cell_size, average_color_block)
            if invert_ascii:
                ascii_img = invert_image(ascii_img)
            color_converted = blend_ascii_with_color(ascii_img, color_blocks, 1)

        if antialiasing:
            color_converted = blend_pixels(ascii_img, color_converted)
        color_converted = copy_black_pixels(ascii_img, color_converted)
        return color_converted, color_blocks, p_cs

def test():
    templates = ShadeArgUtil.get_palette_json('../../resource/palette_files/palette_default_consolab_fast.json')
    for template in templates:
        print(template)

if __name__ == '__main__':
    test()
