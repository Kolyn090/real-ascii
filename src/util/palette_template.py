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
                 override_widths: dict[str, int] | None = None,
                 override_weights: dict[tuple[str, int], float] | None = None):
        self.layer = layer
        self.chars = chars
        self.image_font = image_font
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio
        self.vector_top_k = vector_top_k
        self.match_method = match_method
        self.override_widths = override_widths
        self.override_weights = override_weights

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
            antialiasing=antialiasing
        )

    def create_flow_writer(self, max_workers: int, antialiasing: bool) -> FlowWriter:
        return FlowWriter(
            chars=self.chars,
            char_bound=self.char_bound,
            override_widths=self.override_widths,
            image_font=self.image_font,
            pad=(1, 1),
            flow_match_method='fast',
            binary_threshold=90,
            override_weights=self.override_weights
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
                override_weights[(item["char"], item["width"])] = item["weight"]

        return PaletteTemplate(
            obj["layer"],
            list(dict.fromkeys(c for c in obj["chars"] if c != '\n')),
            ImageFont.truetype(obj["font"], int(obj["font_size"])),
            (obj["char_bound_width"], obj["char_bound_height"]),
            obj["approx_ratio"],
            obj["vector_top_k"],
            obj["match_method"],
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
                f"Override Widths: {self.override_widths}, "
                f"Override Weights: {self.override_weights}")

def test():
    palette_path = '../../resource/palette_files/jx_files/palette_test.json'
    with open(palette_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        name = content["name"]
        templates = content["templates"]
        print(f"Reading palette from {name}.")
        for template in templates:
            palette = PaletteTemplate.read_from_json(template)
            print(palette)

if __name__ == '__main__':
    test()
