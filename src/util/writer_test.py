import os
import cv2
import time
from slicer import Slicer
from arg_util import ShadeArgUtil, TraceArgUtil

def test_char_templates():
    templates = ShadeArgUtil.get_palette_json('../../resource/gradient_char_files/palette_single.json')
    template = templates[0]
    chars = TraceArgUtil.get_chars('ascii')
    template.chars = chars

    max_workers = 16
    save_to_folder = True
    save_folder = 'chars'

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    writer = template.create_writer(max_workers)
    for char_template in writer.char_templates:
        char = char_template.char
        template = char_template.template
        print(char)
        save_path = os.path.join(save_folder, f'char_{ord(char)}.png')
        if save_to_folder:
            cv2.imwrite(save_path, template)
            cv2.imwrite(os.path.join(save_folder, f'bin_{ord(char)}.png'), char_template.template_binary)
            cv2.imwrite(os.path.join(save_folder, f'small_{ord(char)}.png'), char_template.template_small)

def test_match_cells():
    templates = ShadeArgUtil.get_palette_json('../../resource/gradient_char_files/palette_single.json')
    template = templates[0]
    chars = TraceArgUtil.get_chars('ascii')
    template.chars = chars

    max_workers = 16
    factor = 8
    char_bound = (13, 22)
    img_path = '../trace/contour/contour_240_200.png'
    save_folder = 'test'
    resize_method = 'nearest_neighbor'
    save_to_folder = True
    img = cv2.imread(img_path)
    img = TraceArgUtil.resize(resize_method, img, factor)
    h, w = img.shape[:2]

    if save_to_folder:
        os.makedirs(save_folder, exist_ok=True)

    slicer = Slicer()
    cells = slicer.slice(img, char_bound)
    writer = template.create_writer(max_workers)

    test_methods = ['slow', 'optimized', 'fast', 'vector']
    for method in test_methods:
        writer.get_most_similar = writer.get_matching_method(method)
        start = time.perf_counter()
        converted = writer.match_cells(cells, w, h)[0]
        elapsed = time.perf_counter() - start
        print(f"{method} Time: {elapsed:.6f} seconds")
        cv2.imwrite(os.path.join(save_folder, f'{method}_converted.png'), converted)

if __name__ == '__main__':
    test_match_cells()
