import os
import sys
import cv2
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
    resize_factor = 8
    contrast_factor = 1
    thresholds_gamma = 0.13
    sigma_s = 1
    sigma_r = 0.6

    image = cv2.imread("../../resource/f_input/ultraman-nexus.png")
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
        final_img = invert_image(final_img)
        cv2.imwrite(f"jx_files/final_{i}.png", final_img)
        layers.append(p_cts)

    char_weight = get_char_weight(palettes)
    char_weight[' '] = -10000
    stack_test(layers, char_weight, resize_factor * image.shape[:2][1])

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

    print(char_weight)

    horizontals = []
    for y, row_layers in row_table.items():
        print(f"===============y: {y}===================")
        tiling = overlay(row_layers, char_weight, image_width)
        p_cts = [p_ct for p_ct, _, _ in tiling]
        imgs = [p_ct.char_template.img_binary for p_ct in p_cts]
        horizontal = FlowWriter.concat_images_left_to_right(imgs)
        horizontals.append(horizontal)
    final_img = FlowWriter.concat_images_top_to_bottom(horizontals, (255, 255, 255))
    final_img = invert_image(final_img)
    cv2.imwrite("jx_files/final_img.png", final_img)

def stack_test(layers: list[list[PositionalCharTemplate]],
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

    print(char_weight)

    transitional_horizontals: dict[int, list[np.ndarray]] = {i: [] for i in range(len(layers)-1)}
    print(len(transitional_horizontals))
    for y, row_layers in row_table.items():
        print(f"===============y: {y}===================")
        # tilings = stack_overlay_test(list(reversed(row_layers))[:2], char_weight, image_width)  # transitional
        # tilings = stack_overlay_test(list(reversed(row_layers)), char_weight, image_width)
        tilings = stack_overlay_test(row_layers, char_weight, image_width)

        for i in range(len(tilings)):
            tiling = tilings[i]
            p_cts = [p_ct for p_ct, _, _ in tiling]
            imgs = [p_ct.char_template.img_binary for p_ct in p_cts]
            horizontal = FlowWriter.concat_images_left_to_right(imgs)
            transitional_horizontals[i].append(horizontal)

    for idx, horizontals in transitional_horizontals.items():
        final_img = FlowWriter.concat_images_top_to_bottom(horizontals, (255, 255, 255))
        final_img = invert_image(final_img)
        cv2.imwrite(f"jx_files/final_img_{idx}.png", final_img)

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

def stack_overlay(row_layers: list[list[PositionalCharTemplate]],
            char_weight: dict[str, int],
            image_width: int) \
        -> list[tuple[PositionalCharTemplate, int, int]]:
    """
    Stack overlay is a more advanced way of overlay. Rather than applying overlay to
    all layers, it first applies to layers 1 and 2 and get output A, then applies to
    A and 3 and so on...
    :param row_layers:
    :param char_weight:
    :param image_width:
    :return:
    """
    output: list[PositionalCharTemplate] = row_layers[0]
    result: list[tuple[PositionalCharTemplate, int, int]] = []
    for i in range(1, len(row_layers)):
        row_layer = row_layers[i]
        new_row_layers = [output, row_layer]
        overlay_result = overlay(new_row_layers, char_weight, image_width)
        output = [p_ct for p_ct, s, e in overlay_result]
        result = overlay_result

    if len(result) == 0:
        result = build_position_maps([output])[0]

    return result

def stack_overlay_test(row_layers: list[list[PositionalCharTemplate]],
            char_weight: dict[str, int],
            image_width: int) \
        -> list[list[tuple[PositionalCharTemplate, int, int]]]:
    output: list[PositionalCharTemplate] = row_layers[0]
    result: list[list[tuple[PositionalCharTemplate, int, int]]] = []
    for i in range(1, len(row_layers)):
        row_layer = row_layers[i]
        new_row_layers = [output, row_layer]
        overlay_result = overlay(new_row_layers, char_weight, image_width)
        output = [p_ct for p_ct, s, e in overlay_result]
        result.append(overlay_result)

    if len(result) == 0:
        result = [build_position_maps([output])[0]]

    return result

def overlay(row_layers: list[list[PositionalCharTemplate]],
            char_weight: dict[str, int],
            image_width: int) \
        -> list[tuple[PositionalCharTemplate, int, int]]:
    result = []
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]] = build_position_maps(row_layers)
    begin = 0

    layer_weight = {i: i for i in range(len(row_layers))}

    last_best_choice = -1
    while begin <= image_width:
        len_longest_short_img_from_begin = find_len_longest_short_img_from_begin(pos_maps, begin)

        last_indices_spanning_short_imgs = find_last_indices_spanning_short_imgs(pos_maps,
                                                                                 begin,
                                                                                 len_longest_short_img_from_begin)
        best_choice: int = find_best_offset_choice(pos_maps,
                                                  begin,
                                                  char_weight,
                                                  layer_weight,
                                                  last_indices_spanning_short_imgs,
                                                  last_best_choice)  # This is index of layer
        last_best_choice = best_choice
        first_of_best_in_span = get_index_start_from_begin(pos_maps[best_choice], begin)  # This is index of short image
        last_of_best_in_span = last_indices_spanning_short_imgs[best_choice]  # This is index of short image

        if last_of_best_in_span == -1:
            last_of_best_in_span = first_of_best_in_span

        if last_of_best_in_span == -1:
            return result

        new_extend = pos_maps[best_choice][first_of_best_in_span : last_of_best_in_span + 1]
        result.extend(new_extend)

        best_pos_map = pos_maps[best_choice]
        best_end: int = best_pos_map[last_of_best_in_span][2]
        # apply_offset(pos_maps, last_indices_spanning_short_imgs, best_end, False)

        y = pos_maps[0][0][0].top_left[1]
        # if y == 728:
        #     # print("[Offset yes]: ", end='')
        #     copy = make_copy_of_chars_in_range(pos_maps[0], begin, 100)
        #     print_chars(copy)
        #     copy = make_copy_of_chars_in_range(pos_maps[1], begin, 100)
        #     print_chars(copy)

        begin = determine_new_begin(pos_maps, best_end, char_weight)
        diff = begin - best_end
        if diff > 0:
            result.append(make_filler(diff, pos_maps[0][0][0].char_template.char_bound[1], best_end, y))

    return result

def make_filler(width: int, height: int, start: int, y: int) -> tuple[PositionalCharTemplate, int, int]:
    char_template = CharTemplate("filler", None, (width, height),
                                 np.full((height, width), 255, dtype=np.uint8),
                                 to_binary_strong(np.full((height, width), 255, dtype=np.uint8)),
                                 np.full((height, width), 255, dtype=np.uint8),
                                 np.full((height, width), 255, dtype=np.uint8))
    return PositionalCharTemplate(char_template, (start, y)), start, start + width

def determine_new_begin(pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
                        best_end: int,
                        char_weight: dict[str, int]) -> int:
    result = best_end
    highest_char_val = -float('inf')

    for pos_map in pos_maps:
        for p_ct, s, e in pos_map:
            if s >= best_end:
                # Determine the value of char
                char = p_ct.char_template.char
                if char in char_weight:
                    char_val = char_weight[char]
                    if char_val > highest_char_val:
                        highest_char_val = char_val
                        result = s
                break
    return result

def make_copy_of_chars_in_range(pos_map: list[tuple[PositionalCharTemplate, int, int]],
                                begin: int,
                                span_len: int):
    over = begin + span_len
    stored = []
    for p_ct, s, e in pos_map:
        if s >= begin and e <= over:
            stored.append((p_ct, s, e))
    return stored

def print_chars(pos_map: list[tuple[PositionalCharTemplate, int, int]]):
    stored = []
    for p_ct, s, e in pos_map:
        stored.append(f"{p_ct.char_template.char}(x: {p_ct.top_left[0]}, w: {p_ct.char_template.char_bound[0]}, s: {s}, e: {e})")
    stored_join = ",".join(stored)
    print(f"{stored_join}")

def get_index_start_from_begin(pos_map: list[tuple[PositionalCharTemplate, int, int]],
                               begin: int) -> int:
    for i in range(len(pos_map)):
        p_ct, s, t = pos_map[i]
        if s >= begin:
            return i
    return -1

def apply_offset(pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]],
                 last_indices_spanning_short_imgs: list[int],
                 best_end: int,
                 debug=False):
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
            diff_to_best_end = best_end - first_out_span_start

            if debug:
                print(f"diff: {diff_to_best_end}")

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
        if last_span_index == -1:
            continue
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
    last_indices_spanning_short_imgs: list[int],
    last_best_choice: int) -> int:
    high_score = -float('inf')
    result = 0

    debug = []

    for i in range(len(last_indices_spanning_short_imgs)):
        last_index_spanning_short_imgs = last_indices_spanning_short_imgs[i]
        if last_index_spanning_short_imgs == -1:
            continue

        pos_map = pos_maps[i]
        last_short_img = pos_map[last_index_spanning_short_imgs]
        e = last_short_img[2]
        offset_mse = find_offset_mse(pos_maps, begin, e-begin, last_indices_spanning_short_imgs)
        char_weight_sum, chars = calculate_char_weight_sum(char_weight, pos_map, begin, e-begin)
        curr_layer_weight = layer_weight[i]

        coherence_score = 1 if last_best_choice == i else 0
        chars_join = ",".join(chars)
        debug.append((begin, e, f"[{chars_join}]", char_weight_sum, curr_layer_weight, offset_mse, coherence_score))
        # print(f"Begin: {begin}, End: {e}", f"[{chars_join}]", char_weight_sum * 50, curr_layer_weight * 10, -offset_mse, coherence_score * 50)
        choice_score = calculate_choice_score(offset_mse,
                                              char_weight_sum,
                                              curr_layer_weight,
                                              coherence_score)
        if choice_score > high_score:
            high_score = choice_score
            result = i

    y = pos_maps[last_best_choice][0][0].top_left[1]

    # if y == 728:
    #     print("***************")
    #     for item in debug:
    #         print(f"[Begin: {item[0]}, End: {item[1]}, Chars: {item[2]}, Char Weight: {item[3]}, Char Layer: {item[4]}, Coherence: {item[6]}, Offset: {item[5]}]")
    #     print("***************")

    return result

def calculate_choice_score(offset_mse: int,
                           char_weight_sum: int,
                           curr_layer_weight: int,
                           coherence_score) \
    -> float:
    return char_weight_sum * 50 + curr_layer_weight * 150 - offset_mse * 10 + coherence_score * 5

def calculate_char_weight_sum(char_weight: dict[str, int],
                              pos_map: list[tuple[PositionalCharTemplate, int, int]],
                              begin: int,
                              span_len: int) -> tuple[int, list[str]]:
    over = begin + span_len
    result = 0
    chars = []

    for p_ct, s, e in pos_map:
        if s >= begin and e <= over:
            char = p_ct.char_template.char
            if char in char_weight:
                result += char_weight[char]
                chars.append(char)
    return result, chars

if __name__ == '__main__':
    test()
