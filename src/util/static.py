import cv2
import numpy as np

def invert_image(img: np.ndarray) -> np.ndarray:
    # Float image -> assume normalized 0..1
    if img.dtype.kind == "f":
        return 1.0 - img

    # ---- Binary image cases ----
    bw_only = np.all((img == 0) | (img == 255))
    if bw_only:
        return cv2.bitwise_not(img)

    # uint8 binary mask 0/255
    if img.ndim == 2 and img.dtype == np.uint8:
        mn, mx = img.min(), img.max()
        if (mn, mx) in [(0, 255), (0, 1)]:
            return 255 - img if mx == 255 else 1 - img

    # ---- Color image ----
    result = img.copy()
    result[..., :3] = cv2.bitwise_not(img[..., :3])  # keep alpha if exists
    return result

def floor_fill(img: np.ndarray,
               seed_point: tuple[int, int],
               fill_color: int) -> np.ndarray:
    flood_img = img.copy()
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        new_val = (fill_color,)
    else:
        new_val = (fill_color, fill_color, fill_color)
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood_img, mask, seedPoint=seed_point, newVal=new_val)
    return flood_img

def resize_nearest_neighbor(img: np.ndarray,
                            factor: int) -> np.ndarray:
    scale = factor
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    new_size = (new_width, new_height)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    return resized

def resize_bilinear(img: np.ndarray,
                            factor: int) -> np.ndarray:
    scale = factor
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    new_size = (new_width, new_height)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return resized

def increase_contrast(img: np.ndarray,
                      contrast_factor: float,
                      window_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=window_size)
    cl = clahe.apply(l)

    lab_clahe = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return contrast_img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def smooth_colors(img, sigma_s=75, sigma_r=0.4):
    return cv2.edgePreservingFilter(
        img,
        flags=cv2.RECURS_FILTER,
        sigma_s=sigma_s,
        sigma_r=sigma_r
    )

def to_binary_strong(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    binary = np.zeros(img.shape, dtype=np.uint8)
    binary[img == 255] = 255

    # Hard guarantee
    assert binary.ndim == 2

    return binary

def test():
    img_path = '../../resource/f_input/prof.jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    seed_point = (w - 1, h - 1)
    factor = 2

    # img = invert_image(img)
    # img = floor_fill(img, seed_point, 0)
    img = resize_nearest_neighbor(img, factor)

    cv2.imwrite('test.png', img)

if __name__ == '__main__':
    test()
