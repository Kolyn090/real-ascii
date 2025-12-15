# Trace ASCII Filter

The trace image filter includes two steps. The first step generates
the contour images. You will choose one contour image that you want
to use for the next step. The second step generates the Trace ASCII 
Art. The following guides describes the process.

## 📖 Guide 1: Draw Image Contour
1️⃣ `cd` to `src/trace`.

2️⃣ Execute `contour.py`.
**Example**:
```commandline
python contour.py --image_path ../girl_with_pearl_earring.jpg --canny1_min 0 --canny1_max 270 --canny1_step 20 --canny2_min 0 --canny2_max 270 --canny2_step 20 --dilate_iter 1 --erode_iter 0 --gb_sigmaX 0 --gb_size 5 --contrast_factor 16 --contrast_window_size 8
```

**Parameters**

| argument               | help                                                                       |
|------------------------|----------------------------------------------------------------------------|
| --image_path           | The path of the image.                                                     |
| --save_folder          | The folder where the contour images will be saved to.                      |
| --canny1_min           | The minimum value of threshold1 for Canny.                                 |
| --canny1_max           | The maximum value of threshold1 for Canny.                                 |
| --canny1_step          | The step value of threshold1 for Canny.                                    |
| --canny2_min           | The minimum value of threshold2 for Canny.                                 |
| --canny2_max           | The maximum value of threshold2 for Canny.                                 |
| --canny2_step          | The step value of threshold2 for Canny.                                    |
| --contrast_factor      | The contrast factor based on the original image.                           |
| --contrast_window_size | The kernel size of contrast filter.                                        |
| --gb_size              | The kernel size of GaussianBlur.                                           |
| --gb_sigmaX            | The standard deviation of GaussianBlur kernel in the horizontal direction. |
| --kernel_size          | The kernel size of contour function.                                       |
| --dilate_iter          | The number of iterations of dilate.                                        |
| --erode_iter           | The number of iterations of erode.                                         |
| --invert_color         | Set to True to keep the edges black. Otherwise make the edges white.       |

An example of contour image:

<p align="center">
    <img src="../trace/readme_imgs/img0.png" width="400">
</p>

---

## 📖 Guide 2: Trace ASCII Art
1️⃣ `cd` to `src/trace`.

2️⃣ Execute `trace.py`.
**Example**:
```commandline
python trace.py --image_path ./contour/contour_180_260.png --factor 8 --chars file
```

**Japanese Hiragana:**
```commandline
python trace.py --image_path ./contour/contour_240_200.png --factor 8 --chars file --invert_color True --char_bound_height 24 --char_bound_width 22 --font C:/Windows/Fonts/msgothic.ttc --chars_file_path ../trace/chars_file_jp.txt
```

**Parameters**

| argument            | help                                                                                       |
|---------------------|--------------------------------------------------------------------------------------------|
| --image_path        | The path of the image.                                                                     |
| --resize_method     | The image resize method. Check below for available options.                                |
| --save_path         | The directory where the result image will be saved to.                                     |
| --factor            | The resize factor of the new image.                                                        |
| --font              | The font to be used to render the image.                                                   |
| --chars             | The characters you want to use for rendering the image. Check below for available options. |
| --chars_file_path   | The text file of your characters.                                                          |
| --font_size         | The font size.                                                                             |
| --char_bound_width  | The width of one character. We assume each character has the same size.                    |
| --char_bound_height | The height of one character. We assume each character has the same size.                   |
| --invert_color      | Set to True to invert the color of the result image.                                       |

**resize_method**

| code             | help                                          |
|------------------|-----------------------------------------------|
| nearest neighbor | Resize image with nearest neighbor algorithm. |
| bilinear         | Resize image with bilinear algorithm.         |

**chars**

| code  | help                                                                                   |
|-------|----------------------------------------------------------------------------------------|
| ascii | Use all 128 standard ASCII characters as rendering characters.                         |
| file  | Read characters from file `trace/chars_file.txt`. New line character will be excluded. |

An example of trace ascii art image:

<p align="center">
    <img src="../trace/readme_imgs/img1.png" width="400">
</p>

---

⭐ Image Credit: girl_with_pearl_earring (Wikipedia)

<p align="center">
    <img src="../girl_with_pearl_earring.jpg" width="400">
</p>
