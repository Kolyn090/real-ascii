import os
import sys
import cv2
from arg_util import ShadeArgUtil
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
    resize_factor = 2
    contrast_factor = 1
    thresholds_gamma = 0.15
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
        cv2.imwrite(f"jx_files/final_{i}.png", final_img)
        layers.append(p_cts)
    stack(layers)

def stack(layers: list[list[PositionalCharTemplate]]):
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

    horizontals = []
    for y, row_layers in row_table.items():
        tiling = overlay(row_layers)
        p_cts = [p_ct for p_ct, _, _ in tiling]
        imgs = [p_ct.char_template.img for p_ct in p_cts]
        horizontal = FlowWriter.concat_images_left_to_right(imgs)
        horizontals.append(horizontal)
    final_img = FlowWriter.concat_images_top_to_bottom(horizontals, (255, 255, 255))
    cv2.imwrite("jx_files/final_img.png", final_img)

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

# def get_tile(intervals: list[tuple[PositionalCharTemplate, int, int]], x: int) \
#         -> PositionalCharTemplate | None:
#     for tile, s, e in intervals:
#         if s <= x < e:
#             return tile
#     return None
#
# def get_tile_bs(intervals: list[tuple[PositionalCharTemplate, int, int]],
#                 starts: list[int],
#                 x: int) -> PositionalCharTemplate | None:
#     i = bisect.bisect_right(starts, x) - 1
#     if i < 0:
#         return None  # x is before first interval
#
#     tile, s, e = intervals[i]
#     if x < e:
#         return tile
#     return None
#
# def find_tile_set(row_layers: list[list[PositionalCharTemplate]]) \
#     -> list[CharTemplate]:
#     tile_set: set[CharTemplate] = set()
#     for row_layer in row_layers:
#         tile_set.update(set([p_ct.char_template for p_ct in row_layer]))
#     return list(tile_set)
#
# def find_weights(row_layers: list[list[PositionalCharTemplate]]) \
#     -> list[int]:
#     return [i for i in range(len(row_layers))]

# def consensus_tiling(row_layers: list[list[PositionalCharTemplate]], W: int, row_num: int):
#     pos_maps = build_position_maps(row_layers)
#     tile_set = find_tile_set(row_layers)
#     weights = find_weights(row_layers)
#
#     dp = [-float('inf')] * (W+1)
#     choice: list[CharTemplate | None] = [None] * (W + 1)
#     prev: list[int | None] = [None] * (W + 1)
#
#     dp[0] = 0
#     for x in range(W):
#         if dp[x] == -float('inf'):
#             continue
#         for t in tile_set:
#             w = t.char_bound[0]
#             nx = x + w
#             if nx > W:
#                 continue
#
#             gain = 0
#             for seq_map, weight in zip(pos_maps, weights):
#                 segment = seq_map[x:nx]
#
#                 # skip if segment incomplete
#                 if len(segment) != w:
#                     continue
#
#                 # must not contain None
#                 if any(s is None for s in segment):
#                     continue
#
#                 # all must match same tile id
#                 # ids = {tile_id[s] for s in segment}
#                 # if len(ids) == 1 and tid in ids:
#                 #     gain += weight
#
#             score = dp[x] + gain
#
#             # IMPORTANT: allow update even if gain == 0
#             if score > dp[nx]:
#                 dp[nx] = score
#                 prev[nx] = x
#                 choice[nx] = t
#
#     # reconstruct
#     result = []
#     end = max(i for i, p in enumerate(prev) if p is not None or i == 0)
#     x = end
#     while x > 0:
#         t = choice[x]
#         result.insert(0, PositionalCharTemplate(t, (x, row_num * t.char_bound[1])))
#         x = prev[x]
#
#     return result, dp[W]

def overlay(row_layers: list[list[PositionalCharTemplate]]) \
        -> list[tuple[PositionalCharTemplate, int, int]]:
    # One row_layer = one long image
    pos_maps: list[list[tuple[PositionalCharTemplate, int, int]]] = build_position_maps(row_layers)
    pos_map = pos_maps[0]  # Always take everything from the first layer
    count = 1
    while count < len(pos_maps):
        over_layer = pos_maps[count]
        start_set = get_start_set(pos_map)
        end_set = get_end_set(pos_map)
        for p_ct, s, e in over_layer:
            if p_ct.char_template.char != ' ' and s in start_set and e in end_set:
                print(p_ct)
                pos_map = replace(pos_map, s, e, p_ct)

        count += 1
    return pos_map

def get_start_set(pos_map: list[tuple[PositionalCharTemplate, int, int]]) -> list[int]:
    return [s for _, s, _ in pos_map]

def get_end_set(pos_map: list[tuple[PositionalCharTemplate, int, int]]) -> list[int]:
    return [e for _, _, e in pos_map]

def replace(pos_map: list[tuple[PositionalCharTemplate, int, int]],
            start_x: int,
            end_x: int,
            p_ct: PositionalCharTemplate) -> list[tuple[PositionalCharTemplate, int, int]]:
    removed = [tu for tu in pos_map if start_x > tu[1] < end_x]
    removed.append((p_ct, start_x, end_x))
    return removed

if __name__ == '__main__':
    test()
