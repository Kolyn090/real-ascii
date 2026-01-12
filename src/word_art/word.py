import os
import cv2
import sys
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
from static import invert_image  # type: ignore

def trace():
    # Draw contour
    subprocess.run([
        sys.executable,
        "../edge_trace/contour.py",
        "--image_path", "f_output/word.png",
        "--canny1_min", "180",
        "--canny1_max", "181",
        "--canny1_step", "1",
        "--canny2_min", "260",
        "--canny2_max", "261",
        "--canny2_step", "1",
        "--dilate_iter", "1",
        "--erode_iter", "0",
        "--gb_sigmaX", "0",
        "--gb_size", "5",
        "--contrast_factor", "1",
        "--contrast_window_size", "4"
    ])

    # Draw trace ascii art
    subprocess.run([
        sys.executable,
        "../edge_trace/edge_trace.py",
        "--image_path", "./contour/contour_180_260.png",
        "--resize_method", "nearest_neighbor",
        "--resize_factor", "1",
        "--palette_path", "../../resource/palette_files/palette_chars_consolab_fast.json",
        "--match_method", "slow"
    ])

def shade():
    img = cv2.imread("f_output/word.png")
    inverted_img = invert_image(img)
    cv2.imwrite("f_output/inverted_word.png", inverted_img)

    subprocess.run([
        sys.executable,
        "../depth_shade/depth_shade.py",
        "--image_path", "f_output/inverted_word.png",
        "--palette_path", "../../resource/palette_files/palette_default_consolab_fast.json",
        "--resize_factor", "2",
        "--thresholds_gamma", "0.7",
        "--char_weight_sum_factor", "50",
        "--curr_layer_weight_factor", "150",
        "--offset_mse_factor", "10",
        "--coherence_score_factor", "5",
        "--color_option", "original",
        "--antialiasing"
    ])

def main():
    # create_image()
    # trace()
    shade()

def create_image():
    os.makedirs("f_output", exist_ok=True)

    start = (25, 0)
    bound = (1460, 216)
    word = "Real ASCII"
    # font_path = "C:/Windows/Fonts/ariblk.ttf"
    font_path = "C:/Windows/Fonts/consolab.ttf"
    font_size = 256
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("RGB", bound, "white")
    draw = ImageDraw.Draw(img)
    draw.text(start, word, font=font, fill="black")

    img = np.array(img)

    cv2.imwrite("f_output/word.png", img)

if __name__ == '__main__':
    main()
