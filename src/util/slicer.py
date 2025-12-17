import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Cell:
    def __init__(self, top_left: tuple[int, int], img: np.ndarray):
        self.top_left = top_left
        self.img = img

class Slicer:
    def __init__(self, max_workers=16):
        self.max_workers = max_workers

    def slice(self, img: np.ndarray, cell_size: tuple[int, int]) -> list[Cell]:
        h, w = img.shape[:2]
        cell_w, cell_h = cell_size
        tasks = []
        # Build slicing tasks (fast + no threading yet)
        for i in range(-1, w, cell_w):
            for j in range(-1, h, cell_h):
                top_left = (i + 1, j + 1)
                bottom_right = (i + cell_w, j + cell_h)

                if not self.in_range(top_left, bottom_right, w, h):
                    continue

                tasks.append((img, top_left, bottom_right))

        # Run multithreaded slicing
        result = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for top_left, crop in executor.map(self.slice_one, tasks):
                cell = Cell(top_left, crop)
                result.append(cell)

        return result

    @staticmethod
    def slice_one(args):
        img, top_left, bottom_right = args
        x1, y1 = top_left
        x2, y2 = bottom_right

        crop = img[y1:y2 + 1, x1:x2 + 1]
        return top_left, crop

    @staticmethod
    def in_range(top_left, bottom_right, w, h):
        x1, y1 = top_left
        x2, y2 = bottom_right

        return (
                0 <= x1 < w and
                0 <= y1 < h and
                0 <= x2 < w and
                0 <= y2 < h
        )
