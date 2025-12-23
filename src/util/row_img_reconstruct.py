import numpy as np
from writer import CharTemplate

def reconstruct2(long_img: np.ndarray, char_templates: list[CharTemplate]) \
        -> tuple[list[CharTemplate], float] | None:
    long_img = long_img.astype(bool)
    # key = index
    # value = (char template, short image)
    ct_table: dict[int, tuple[CharTemplate, np.ndarray]] = dict()
    for i in range(len(char_templates)):
        short_img = char_templates[i].img_binary.astype(bool)
        ct_table[i] = (char_templates[i], short_img)

    H, W = long_img.shape

    # --- DP arrays ---
    dp = [float('inf')] * (W + 1)
    choice: list[CharTemplate | None] = [None] * (W + 1)
    prev: list[int | None] = [None] * (W + 1)
    dp[0] = 0

    # --- DP ---
    for x in range(W + 1):
        if dp[x] == float('inf'):
            continue

        for i in range(len(ct_table)):
            tile = ct_table[i][1]
            _, w = tile.shape
            nx = min(x + w, W)
            region = long_img[:, x:nx]
            tile_crop = tile[:, :region.shape[1]]
            c = np.count_nonzero(region != tile_crop)

            if dp[x] + c < dp[nx]:
                dp[nx] = dp[x] + c
                choice[nx] = ct_table[i][0]
                prev[nx] = x

    # --- reconstruct solution ---
    seq: list[CharTemplate] = []

    x = W
    while x > 0:
        seq.insert(0, choice[x])
        x = prev[x]

    return seq, dp[W]

def reconstruct(long_img: np.ndarray, short_imgs: list[np.ndarray]) -> tuple[list[np.ndarray], float] | None:
    long_img = long_img.astype(bool)
    short_imgs = [t.astype(bool) for t in short_imgs]

    H, W = long_img.shape

    # --- DP arrays ---
    dp = [float('inf')] * (W + 1)
    choice: list[np.ndarray | None] = [None] * (W + 1)
    prev: list[int | None] = [None] * (W + 1)
    dp[0] = 0

    # --- DP ---
    for x in range(W + 1):
        if dp[x] == float('inf'):
            continue

        for tile in short_imgs:
            _, w = tile.shape
            nx = min(x + w, W)  # truncate at end
            region = long_img[:, x:nx]
            tile_crop = tile[:, :region.shape[1]]
            c = np.count_nonzero(region != tile_crop)

            if dp[x] + c < dp[nx]:
                dp[nx] = dp[x] + c
                choice[nx] = tile_crop
                prev[nx] = x

    # --- reconstruct solution ---
    seq: list[np.ndarray] = []

    x = W
    while x > 0:
        seq.insert(0, choice[x])
        x = prev[x]

    return seq, dp[W]
