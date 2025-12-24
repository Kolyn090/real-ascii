from writer import Writer
from flow_writer import FlowWriter
from PIL.ImageFont import FreeTypeFont

class PaletteTemplate:
    def __init__(self,
                 layer: int,
                 chars: list[str],
                 image_font: FreeTypeFont,
                 char_bound: tuple[int, int],
                 approx_ratio: float,
                 vector_top_k: int,
                 match_method: str,
                 override_widths: dict[str, int] | None = None):
        self.layer = layer
        self.chars = chars
        self.image_font = image_font
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio
        self.vector_top_k = vector_top_k
        self.match_method = match_method
        self.override_widths = override_widths

    def create_writer(self, max_workers: int) -> Writer:
        return Writer(
            image_font=self.image_font,
            max_workers=max_workers,
            char_bound=self.char_bound,
            approx_ratio=self.approx_ratio,
            match_method=self.match_method,
            vector_top_k=self.vector_top_k,
            chars=self.chars,
            override_widths=self.override_widths
        )

    def create_flow_writer(self, max_workers: int) -> FlowWriter:
        return FlowWriter(
            chars=self.chars,
            char_bound=self.char_bound,
            override_widths=self.override_widths,
            image_font=self.image_font,
            gap=1,
            flow_match_method='fast',
            binary_threshold=90
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
                f"Override Widths: {self.override_widths}")
