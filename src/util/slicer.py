import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Cell:
    def __init__(self):
        self.top_left: tuple[int, int] | None = None
        self.img: np.ndarray | None = None

class Slicer:
    def __init__(self):
        self.max_workers = 16

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
                cell = Cell()
                cell.top_left = top_left
                cell.img = crop
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

def main():
    img_path = '../binary/bin_85.png'
    img = cv2.imread(img_path)
    save_to_folder = False
    save_folder = 'test'
    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    slicer = Slicer()
    cells = slicer.slice(img, (13, 22))
    for cell in cells:
        top_left = cell.top_left
        crop = cell.img
        print(top_left)
        save_path = os.path.join(save_folder, f"slice_{top_left[0]}_{top_left[1]}.png")
        if save_to_folder:
            cv2.imwrite(save_path, crop)

if __name__ == '__main__':
    main()
