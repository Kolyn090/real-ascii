from writer import Writer
from PIL.ImageFont import FreeTypeFont

class PaletteTemplate:
    def __init__(self,
                 layer: int,
                 chars: list[str],
                 imageFont: FreeTypeFont,
                 char_bound: tuple[int, int],
                 approx_ratio: float,
                 vector_top_k: int,
                 match_method: str):
        self.layer = layer
        self.chars = chars
        self.imageFont = imageFont
        self.char_bound = char_bound
        self.approx_ratio = approx_ratio
        self.vector_top_k = vector_top_k
        self.match_method = match_method

    def create_writer(self, max_workers) -> Writer:
        return Writer(
            imageFont=self.imageFont,
            max_workers=max_workers,
            char_bound=self.char_bound,
            approx_ratio=self.approx_ratio,
            match_method=self.match_method,
            vector_top_k=self.vector_top_k,
            chars=self.chars
        )

    def __str__(self):
        chars = "".join(self.chars)
        return (f"Layer: {self.layer}, "
                f"Font: {self.imageFont.getname()}, "
                f"Character Bound: {self.char_bound}, "
                f"Approximate Ratio: {self.approx_ratio}, "
                f"Vector Top K: {self.vector_top_k}, "
                f"Method: {self.match_method}, "
                f"Chars: {chars} ")
