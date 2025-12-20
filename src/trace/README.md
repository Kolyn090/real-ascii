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
python contour.py --image_path ../../resource/imgs/girl_with_pearl_earring.jpg --canny1_min 0 --canny1_max 270 --canny1_step 20 --canny2_min 0 --canny2_max 270 --canny2_step 20 --dilate_iter 1 --erode_iter 0 --gb_sigmaX 0 --gb_size 5 --contrast_factor 16 --contrast_window_size 8
```

**Another Example**:
```commandline
python contour.py --image_path ../../resource/imgs/tsunami.jpg --canny1_min 0 --canny1_max 270 --canny1_step 20 --canny2_min 0 --canny2_max 270 --canny2_step 20 --dilate_iter 1 --erode_iter 0 --gb_sigmaX 0 --gb_size 5 --contrast_factor 4 --contrast_window_size 8
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
| --invert_color         | If included, paint the edges white and the background black.               |

An example of contour image:

<p align="center">
    <img src="../../resource/readme_imgs/trace_girl.png" width="400">
</p>

---

## 📖 Guide 2: Trace ASCII Art
1️⃣ `cd` to `src/trace`.

2️⃣ Execute `trace.py`.
**Example**:
```commandline
python trace.py --image_path ./contour/contour_180_260.png --resize_factor 8 --chars file --font C:/Windows/Fonts/consolab.ttf --char_bound_width 13 --char_bound_height 22 --match_method slow 
```

**Japanese Hiragana**:
```commandline
python trace.py --image_path ./contour/contour_240_200.png --resize_factor 8 --chars file --char_bound_height 24 --char_bound_width 22 --font C:/Windows/Fonts/msgothic.ttc --font_size 24 --chars_file_path ../../resource/char_files/chars_file_hiragana.txt --match_method vector --approx_ratio 0.5 --vector_top_k 5 --invert_color
```

**An Example using Palette file**, Check out the [palette tutorial](../../palette_tut.md) if you would like to make your own palette:
```commandline
python trace.py --image_path ./contour/contour_180_260.png --resize_factor 8 --palette_path ../../resource/palette_files/palette_chars.json --match_method slow
```

**Parameters**

| argument              | help                                                                                                                   |
|-----------------------|------------------------------------------------------------------------------------------------------------------------|
| --image_path          | The path of the image.                                                                                                 |
| --save_path           | The directory where the result image will be saved to.                                                                 |
| --resize_method       | The image resize method. Check below for available options.                                                            |
| --resize_factor       | The resize factor of the new image.                                                                                    |
| --font                | The font to be used to render the image.                                                                               |
| --chars               | The characters you want to use for rendering the image. Check below for available options.                             |
| --chars_file_path     | The text file of your characters.                                                                                      |
| --font_size           | The font size.                                                                                                         |
| --char_bound_width    | The width of one character. We assume each character has the same size.                                                |
| --char_bound_height   | The height of one character. We assume each character has the same size.                                               |
| --invert_color        | If included, invert the color of the result image.                                                                     |
| --max_workers         | The maximum number of multithread workers.                                                                             |
| --match_method        | The algorithm for template (character) matching. Check below for available options.                                    |
| --approx_ratio        | Only used if the matching method is 'vector'. Each smaller image will be resized by this value before comparison.      |
| --vector_top_k        | Only used if the matching method is 'vector'. Only compare the smaller image to the k best candidates.                 |
| --palette_path        | Use a palette. Only the first template will be used. The values in template can be overridden with explicit arguments. |
| --color_option        | The option to color the image. Check below for available options.                                                      |
| --original_image_path | REQUIRED if you are doing `color_option=original`.                                                                     |
| --save_chars          | If included, the characters will be saved to a file.                                                                   |
| --save_chars_path     | The path to save the characters. Check out the 'f_output' folder for the results.                                      |

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

**match_method**

| code      | help                                                                                                                                |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| slow      | The slowest matching algorithm with the best matching quality. The templates are grayscale.                                         |
| optimized | Almost twice as fast as slow. The templates are binary. The resulting image will look bold compared to slow method.                 |
| fast      | Almost twice as fast as optimized. Utilizes XOR comparison. The resulting image is very similar to optimized method.                |
| vector    | Almost ten times as fast as slow. Vectorize all smaller images and compare the flattened array. The resulting image is much bolder. |

**color_option**

| code     | help                                                           |
|----------|----------------------------------------------------------------|
| original | Color the ASCII art with the (resized) original image's color. |

An example of ascii art image:

<p align="center">
    <img src="../../resource/readme_imgs/trace_ascii_girl.png" width="400">
</p>

---

## 📕 Extra Guide: Join-Trace ASCII Art

1️⃣ `cd` to `src/trace`.

2️⃣ Execute `trace_join.py`.

**Example**:
```commandline
python trace_join.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--canny1 180 ^
--canny2 260 ^
--gb_size 5 ^
--gb_sigmaX 0 ^
--kernel_size 2 ^
--dilate_iter 1 ^
--erode_iter 0 ^
--contrast_factor 4 ^
--contrast_window_size 8 ^
--resize_factor 8 ^
--resize_method nearest_neighbor ^
--match_method slow ^
--palette_path ../../resource/palette_files/palette_chars.json
```

**Parameters**

| argument               | help                                                                                                                   |
|------------------------|------------------------------------------------------------------------------------------------------------------------|
| --image_path           | The path of the image.                                                                                                 |
| --canny1               | The value of threshold1 for Canny.                                                                                     |
| --canny2               | The value of threshold2 for Canny.                                                                                     |
| --gb_size              | The kernel size of GaussianBlur.                                                                                       |
| --gb_sigmaX            | The standard deviation of GaussianBlur kernel in the horizontal direction.                                             |
| --kernel_size          | The kernel size of contour function.                                                                                   |
| --dilate_iter          | The number of iterations of dilate.                                                                                    |
| --erode_iter           | The number of iterations of erode.                                                                                     |
| --contrast_factor      | The contrast factor based on the original image.                                                                       |
| --contrast_window_size | The kernel size of contrast filter.                                                                                    |
| --save_path            | The directory where the result image will be saved to.                                                                 |
| --resize_method        | The image resize method. Check below for available options.                                                            |
| --resize_factor        | The resize factor of the new image.                                                                                    |
| --font                 | The font to be used to render the image.                                                                               |
| --chars                | The characters you want to use for rendering the image. Check below for available options.                             |
| --chars_file_path      | The text file of your characters.                                                                                      |
| --font_size            | The font size.                                                                                                         |
| --char_bound_width     | The width of one character. We assume each character has the same size.                                                |
| --char_bound_height    | The height of one character. We assume each character has the same size.                                               |
| --invert_color         | If included, invert the color of the result image.                                                                     |
| --max_workers          | The maximum number of multithread workers.                                                                             |
| --match_method         | The algorithm for template (character) matching. Check below for available options.                                    |
| --approx_ratio         | Only used if the matching method is 'vector'. Each smaller image will be resized by this value before comparison.      |
| --vector_top_k         | Only used if the matching method is 'vector'. Only compare the smaller image to the k best candidates.                 |
| --palette_path         | Use a palette. Only the first template will be used. The values in template can be overridden with explicit arguments. |
| --color_option         | The option to color the image. Check below for available options.                                                      |
| --save_chars           | If included, the characters will be saved to a file.                                                                   |
| --save_chars_path      | The path to save the characters. Check out the 'f_output' folder for the results.                                      |

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

**match_method**

| code      | help                                                                                                                                |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| slow      | The slowest matching algorithm with the best matching quality. The templates are grayscale.                                         |
| optimized | Almost twice as fast as slow. The templates are binary. The resulting image will look bold compared to slow method.                 |
| fast      | Almost twice as fast as optimized. Utilizes XOR comparison. The resulting image is very similar to optimized method.                |
| vector    | Almost ten times as fast as slow. Vectorize all smaller images and compare the flattened array. The resulting image is much bolder. |

**color_option**

| code     | help                                                           |
|----------|----------------------------------------------------------------|
| original | Color the ASCII art with the (resized) original image's color. |

An example of ascii art image:

<p align="center">
    <img src="../../resource/readme_imgs/trace_join_girl.png" width="400">
</p>

---

🖼️ Also check out the [gallery](./gallery.md) for more examples!

---

⭐ Image Credit: girl_with_pearl_earring (Wikipedia)

<p align="center">
    <img src="../../resource/imgs/girl_with_pearl_earring.jpg" width="400">
</p>
