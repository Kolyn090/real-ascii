from PIL import Image

def pil_pad_columns(img: Image.Image, num: int, color="white") -> Image.Image:
    num = max(num, 0)
    w, h = img.size
    new_img = Image.new(img.mode, (w + 2 * num, h), color)
    new_img.paste(img, (num, 0))
    return new_img

def pil_pad_rows(img: Image.Image, num: int, color="white") -> Image.Image:
    num = max(num, 0)
    w, h = img.size
    new_img = Image.new(img.mode, (w, h + 2 * num), color)
    new_img.paste(img, (num, 0))
    return new_img
