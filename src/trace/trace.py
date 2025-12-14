import os.path
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from slicer import Slicer  # type: ignore
from writer import Writer  # type: ignore
from static import invert_image, floor_fill, resize_nearest_neighbor  # type: ignore

def main():
    factor = 4
    img_path = './test/contour_35_100.png'
    save_path = 'ascii_art.png'
    img = cv2.imread(img_path)
    img = resize_nearest_neighbor(img, factor)

    h, w = img.shape[:2]
    seed_point = (w - 1, h - 1)
    chars = [chr(i) for i in range(128)]

    slicer = Slicer()
    cells = slicer.slice(img, (13, 22))
    writer = Writer()
    writer.assign_char_templates(chars)
    converted = writer.match_cells(cells, w, h)[0]
    converted = invert_image(converted)
    converted = floor_fill(converted, seed_point, 0)
    cv2.imwrite(save_path, converted)

if __name__ == '__main__':
    main()
