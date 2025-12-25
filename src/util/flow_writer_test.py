import os
import sys
import cv2
import math
import numpy as np
from arg_util import ShadeArgUtil
from palette_template import PaletteTemplate
from static import (to_binary_strong, to_grayscale, increase_contrast,  # type: ignore
                    resize_nearest_neighbor, to_binary_middle, smooth_colors,
                    invert_image)  # type: ignore
from writer import CharTemplate  # type: ignore
from flow_writer import FlowWriter
from char_template import PositionalCharTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shade')))
from gradient_divide import divide  # type: ignore

def test():
    max_workers = 16
    resize_factor = 4
    contrast_factor = 1
    thresholds_gamma = 0.15
    sigma_s = 1
    sigma_r = 0.6

    image = cv2.imread("../../resource/imgs/monalisa.jpg")
    image = resize_nearest_neighbor(image, resize_factor)
    image = increase_contrast(image, contrast_factor)
    image = smooth_colors(image, sigma_s, sigma_r)
    image = to_grayscale(image)

    palettes = ShadeArgUtil.get_palette_json('../../resource/palette_files/jx_files/palette_test.json')
    # palette = palettes[1]
    # flow_writer = palette.create_flow_writer(max_workers)
    # final_img, p_cts = flow_writer.match(image)
    # cv2.imwrite("final_img.png", final_img)

    gradient_imgs = divide(image, len(palettes), thresholds_gamma)
    # os.makedirs("jx_files", exist_ok=True)
    # count = 0
    # for gradient_img in gradient_imgs:
    #     cv2.imwrite(f"jx_files/gradient_{count}.png", gradient_img)
    #     count += 1

    layers = []
    for i in range(len(palettes)):
        palette = palettes[i]
        gradient_img = gradient_imgs[i]
        gradient_img = invert_image(gradient_img)
        flow_writer = palette.create_flow_writer(max_workers)
        final_img, p_cts = flow_writer.match(gradient_img)
        cv2.imwrite(f"jx_files/final_{i}.png", final_img)
        layers.append(p_cts)

    char_weight = get_char_weight(palettes)
    stack(layers, char_weight, resize_factor * image.shape[:2][1])

def get_char_weight(palettes: list[PaletteTemplate]) -> dict[str, int]:
    result = dict()

    for i in range(len(palettes)):
        for j in range(len(palettes[i].chars)):
            char = palettes[i].chars[j]
            if char not in result:
                result[char] = j * 2 + i

    return result

def stack(layers: list[list[PositionalCharTemplate]],
          char_weight: dict[str, int],
          image_width: int):
    row_table: dict[int, list[list[PositionalCharTemplate]]] = dict()
    for i in range(len(layers)):
        layer = layers[i]
        for p_ct in layer:
            y = p_ct.top_left[1]
            if y not in row_table:
                row_table[y] = []
            if len(row_table[y]) < i+1:
                row_table[y].append([])
            row_table[y][i].append(p_ct)

    # row0 = row_table[0]
    # overlayed = overlay(row0, char_weight, image_width)
    # for p_ct, s, e in overlayed:
    #     print(p_ct, s ,e)

    print(char_weight)

    horizontals = []
    for y, row_layers in row_table.items():
        print(f"===============y: {y}===================")
        tiling = overlay(row_layers, char_weight, image_width)
        p_cts = [p_ct for p_ct, _, _ in tiling]
        imgs = [p_ct.char_template.img for p_ct in p_cts]
        horizontal = FlowWriter.concat_images_left_to_right(imgs)
        horizontals.append(horizontal)
    final_img = FlowWriter.concat_images_top_to_bottom(horizontals, (255, 255, 255))
    final_img = invert_image(final_img)
    cv2.imwrite("jx_files/final_img.png", final_img)

    # count = 0
    # final_rows = []
    # for y, row_layers in row_table.items():
    #     horizontals = []
    #     for i in range(len(row_layers)):
    #         p_cts = row_layers[i]
    #         imgs = [p_ct.char_template.img for p_ct in p_cts]
    #         horizontal = FlowWriter.concat_images_left_to_right(imgs)
    #         horizontals.append(horizontal)
    #     final_row_img = merge_nonwhite(horizontals, 255)
    #     final_rows.append(final_row_img)
    #     count += 1
    # final_img = FlowWriter.concat_images_top_to_bottom(final_rows, (255, 255, 255))
    # final_img = invert_image(final_img)
    # cv2.imwrite(f"jx_files/final_img.png", final_img)

def merge_nonwhite(images, fill_value=255):
    if not images:
        raise ValueError("No images given")

    H = images[0].shape[0]
    is_rgb = (images[0].ndim == 3)

    for img in images:
        if img.shape[0] != H:
            raise ValueError("Images must have same height")
        if (img.ndim == 3) != is_rgb:
            raise ValueError("Don't mix grayscale and RGB images")

    max_w = max(img.shape[1] for img in images)

    if is_rgb:
        merged = np.full((H, max_w, 3), fill_value, dtype=np.uint8)
        for img in images:
            w = img.shape[1]
            merged[:, :w] = np.minimum(merged[:, :w], img)
    else:
        merged = np.full((H, max_w), fill_value, dtype=np.uint8)
        for img in images:
            w = img.shape[1]
            merged[:, :w] = np.minimum(merged[:, :w], img)

    return merged

# Caution: the following code only take care of one row!

def build_position_maps(row_layers: list[list[PositionalCharTemplate]]) \
        -> list[list[tuple[PositionalCharTemplate, int, int]]]:
    """
    For each row layer, convert the short images to
    tuples of (short image, start x, end x)
    :param row_layers:
    :return:
    """
    result = []
    for layer in row_layers:
        curr = 0
        intervals = []
        for p_ct in layer:
            w = p_ct.char_template.char_bound[0]
            intervals.append((p_ct, curr, curr + w))
            curr += w
        result.append(intervals)
    return result

# def overlay(row_layers: list[list[PositionalCharTemplate]]) \
#         -> list[tuple[PositionalCharTemplate, int, int]]:
#     # One row_layer = one long image
#     pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]] = build_position_maps(row_layers)
#     pos_map = pos_maps[0]  # Always take everything from the first layer
#     count = 1
#     while count < len(pos_maps):
#         over_layer = pos_maps[count]
#         start_set = get_start_set(pos_map)
#         end_set = get_end_set(pos_map)
#         for p_ct, s, e in over_layer:
#             if p_ct.char_template.char != ' ' and s in start_set and e in end_set:
#                 print(p_ct)
#                 pos_map = replace(pos_map, s, e, p_ct)
#
#         count += 1
#     return pos_map
#
# def get_start_set(pos_map: list[tuple[PositionalCharTemplate, int, int]]) -> list[int]:
#     return [s for _, s, _ in pos_map]
#
# def get_end_set(pos_map: list[tuple[PositionalCharTemplate, int, int]]) -> list[int]:
#     return [e for _, _, e in pos_map]
#
# def replace(pos_map: list[tuple[PositionalCharTemplate, int, int]],
#             start_x: int,
#             end_x: int,
#             p_ct: PositionalCharTemplate) -> list[tuple[PositionalCharTemplate, int, int]]:
#     removed = [tu for tu in pos_map if start_x > tu[1] < end_x]
#     removed.append((p_ct, start_x, end_x))
#     return removed

def overlay(row_layers: list[list[PositionalCharTemplate]],
            char_weight: dict[str, int],
            image_width: int) \
        -> list[tuple[PositionalCharTemplate, int, int]]:
    result = []
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]] = build_position_maps(row_layers)
    begin = 0

    layer_weight = {i: i for i in range(len(row_layers))}

    while begin <= image_width:
        len_longest_short_img_from_begin = find_len_longest_short_img_from_begin(pos_maps, begin)
        last_indices_spanning_short_imgs = find_last_indices_spanning_short_imgs(pos_maps,
                                                                                 begin,
                                                                                 len_longest_short_img_from_begin)
        best_choice: int = find_best_offset_choice(pos_maps,
                                              begin,
                                              char_weight,
                                              layer_weight,
                                              last_indices_spanning_short_imgs)  # This is index of layer
        first_of_best_in_span = get_index_start_from_begin(pos_maps[best_choice], begin)  # This is index of short image
        last_of_best_in_span = last_indices_spanning_short_imgs[best_choice]  # This is index of short image

        if last_of_best_in_span == -1:
            return result

        result.extend(pos_maps[best_choice][first_of_best_in_span : last_of_best_in_span + 1])

        best_pos_map = pos_maps[best_choice]
        best_end: int = best_pos_map[last_of_best_in_span][2]
        apply_offset(pos_maps, last_indices_spanning_short_imgs, best_end)
        begin = best_end

    return result

def get_index_start_from_begin(pos_map: list[tuple[PositionalCharTemplate, int, int]],
                               begin: int) -> int:
    for i in range(len(pos_map)):
        p_ct, s, t = pos_map[i]
        if s >= begin:
            return i
    return -1

def apply_offset(pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
                 last_indices_spanning_short_imgs: list[int],
                 best_end: int):
    """

    :param pos_maps:
    :param last_indices_spanning_short_imgs:
    :param best_end:
    :return:
    """

    """
    | ... | spanning images | next |
    we want to shift 'next' to left or to right so that its 'start' will
    align with the 'end' of our best spanning images
    """

    for i in range(len(last_indices_spanning_short_imgs)):
        last_index_spanning_short_imgs = last_indices_spanning_short_imgs[i]
        if last_index_spanning_short_imgs == -1:
            continue

        pos_map = pos_maps[i]
        next_last = last_index_spanning_short_imgs + 1
        if next_last < len(pos_map):
            first_out_span = pos_map[next_last]
            first_out_span_start = first_out_span[1]
            diff_to_best_end = first_out_span_start - best_end

            # print(f"Layer: {i}, Start: {first_out_span_start}, Next Last: {next_last}")

            # Now adding 'diff_to_best_end' to all (start, end) in out span
            for j in range(next_last, len(pos_map)):
                p_ct, s, e = pos_map[j]
                pos_map[j] = (p_ct, s + diff_to_best_end, e + diff_to_best_end)

def find_len_longest_short_img_from_begin(
        pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
        begin: int) -> int:
    # Assume each layer has exactly one short img starting from 'begin'
    longest_len = 0

    for pos_map in pos_maps:
        for p_ct, s, e in pos_map:
            if s == begin:
                w = p_ct.char_template.char_bound[0]
                if w > longest_len:
                    longest_len = w
                break
    return longest_len

def find_last_indices_spanning_short_imgs(
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
    begin: int,
    span_len: int) -> list[int]:
    result = []
    over = begin + span_len

    for i in range(len(pos_maps)):
        last = -1
        for j in range(len(pos_maps[i])):
            p_ct, s, e = pos_maps[i][j]
            if begin <= s and e <= over:
                last = j
        result.append(last)
    return result

def find_offset_mse(
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
    begin: int,
    span_len: int,
    last_span_indices: list[int]) -> int:
    result = 0
    over = begin + span_len

    for i in range(len(last_span_indices)):
        last_span_index = last_span_indices[i]
        pos_map = pos_maps[i]
        if last_span_index != -1:
            next_last = last_span_index + 1
            if next_last < len(pos_map):
                next_last_short_img = pos_map[next_last]
                s = next_last_short_img[1]
                result += (s - over) * (s - over)
    return result

def find_best_offset_choice(
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
    begin: int,
    char_weight: dict[str, int],
    layer_weight: dict[int, int],
    last_indices_spanning_short_imgs: list[int]) -> int:
    high_score = -float('inf')
    result = 0

    for i in range(len(last_indices_spanning_short_imgs)):
        last_index_spanning_short_imgs = last_indices_spanning_short_imgs[i]
        pos_map = pos_maps[i]
        last_short_img = pos_map[last_index_spanning_short_imgs]
        e = last_short_img[2]
        offset_mse = find_offset_mse(pos_maps, begin, e-begin, last_indices_spanning_short_imgs)
        char_weight_sum = calculate_char_weight_sum(char_weight, pos_map, begin, e-begin)
        curr_layer_weight = layer_weight[i]
        choice_score = calculate_choice_score(offset_mse, char_weight_sum, curr_layer_weight)
        # print(math.log2(offset_mse) if offset_mse != 0 else 0, char_weight_sum * 10, curr_layer_weight * 10)
        if choice_score > high_score:
            high_score = choice_score
            result = i
    return result

def calculate_choice_score(offset_mse: int, char_weight_sum: int, curr_layer_weight: int) \
    -> float:
    return char_weight_sum * 5 + curr_layer_weight * 3 - offset_mse

def calculate_char_weight_sum(char_weight: dict[str, int],
                              pos_map: list[tuple[PositionalCharTemplate, int, int]],
                              begin: int,
                              span_len: int) -> int:
    over = begin + span_len
    result = 0

    for p_ct, s, e in pos_map:
        if s >= begin and e <= over:
            char = p_ct.char_template.char
            if char in char_weight:
                result += char_weight[char]
    return result

if __name__ == '__main__':
    test()
