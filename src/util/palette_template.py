import os
import sys
import json
from writer import Writer
from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../nonfixed_width')))
from flow_writer import FlowWriter  # type: ignore

class PaletteTemplate:
    def __init__(self,
                 layer: int,
                 chars: list[str],
                 image_font: FreeTypeFont,
                 char_bound: tuple[int, int],
                 approx_ratio: float,
                 vector_top_k: int,
                 match_method: str,
                 pad: tuple[int, int],
                 override_widths: dict[str, int] | None = None,
                 override_weights: dict[str, float] | None = None):
        self.layer = layer
        self.chars = chars
        self.image_font = image_font
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio
        self.vector_top_k = vector_top_k
        self.match_method = match_method
        self.override_widths = override_widths
        self.override_weights = override_weights
        self.pad = pad

    def create_writer(self, max_workers: int, antialiasing: bool) -> Writer:
        return Writer(
            image_font=self.image_font,
            max_workers=max_workers,
            char_bound=self.char_bound,
            approx_ratio=self.approx_ratio,
            match_method=self.match_method,
            vector_top_k=self.vector_top_k,
            chars=self.chars,
            override_widths=self.override_widths,
            override_weights=self.override_weights,
            antialiasing=antialiasing,
            pad=self.pad
        )

    def create_flow_writer(self, max_workers: int, antialiasing: bool) -> FlowWriter:
        return FlowWriter(
            chars=self.chars,
            char_bound=self.char_bound,
            override_widths=self.override_widths,
            image_font=self.image_font,
            pad=self.pad,
            flow_match_method='fast',
            binary_threshold=90,
            override_weights=self.override_weights,
            antialiasing=antialiasing,
            max_workers=max_workers
        )

    @staticmethod
    def read_from_json(obj: dict):
        override_widths = None
        if "override_widths" in obj:
            override_widths = dict()
            for item in obj["override_widths"]:
                override_widths[item["char"]] = item["width"]

        override_weights = None
        if "override_weights" in obj:
            override_weights = dict()
            for item in obj["override_weights"]:
                override_weights[item["char"]] = item["weight"]

        return PaletteTemplate(
            obj["layer"],
            list(dict.fromkeys(c for c in obj["chars"] if c != '\n')),
            ImageFont.truetype(obj["font"], int(obj["font_size"])),
            (obj["char_bound_width"], obj["char_bound_height"]),
            obj["approx_ratio"],
            obj["vector_top_k"],
            obj["match_method"],
            (obj["pad_width"], obj["pad_height"]),
            override_widths,
            override_weights
        )

    def __str__(self):
        chars = "".join(self.chars)
        return (f"Layer: {self.layer}, "
                f"Font: {self.image_font.getname()}, "
                f"Character Bound: {self.char_bound}, "
                f"Approximate Ratio: {self.approx_ratio}, "
                f"Vector Top K: {self.vector_top_k}, "
                f"Method: {self.match_method}, "
                f"Chars: {chars}, "
                f"Pad: {self.pad}, "
                f"Override Widths: {self.override_widths}, "
                f"Override Weights: {self.override_weights}")

def validate_palettes(palettes: list[PaletteTemplate]):
    # 1. Make sure all characters have the same valid cell height
    expected_char_bound_height = palettes[0].char_bound[1]
    expected_char_bound_height += palettes[0].pad[1] * 2
    for i in range(1, len(palettes)):
        palette = palettes[i]
        char_bound_height = palette.char_bound[1]
        char_bound_height += palette.pad[1] * 2
        if expected_char_bound_height != char_bound_height:
            raise Exception("Not all characters have the same valid cell height!")

    # Add more rules in the future...

    print("All tests passed for palettes.")

def test():
    palette_path = '../../resource/palette_files/jx_files/palette_test.json'
    with open(palette_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        name = content["name"]
        templates = content["templates"]
        print(f"Reading palette from {name}.")
        palettes = []
        for template in templates:
            palette = PaletteTemplate.read_from_json(template)
            palettes.append(palette)
            print(palette)

    validate_palettes(palettes)

def test_invalid():
    palette_path = '../../resource/palette_files/jx_files/palette_invalid.json'
    with open(palette_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        name = content["name"]
        templates = content["templates"]
        print(f"Reading palette from {name}.")
        palettes = []
        for template in templates:
            palette = PaletteTemplate.read_from_json(template)
            palettes.append(palette)

    validate_palettes(palettes)

if __name__ == '__main__':
    test()
