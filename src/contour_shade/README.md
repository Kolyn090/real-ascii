# Contour Shade ASCII Filter

This filter renders shading around the edges in the given image.
Like shade filter, a palette is required. (Might also require more
manual finetuning than the shade filter)

## 📖 Guide: Contour Shade ASCII Art
1️⃣ `cd` to `src/contour_shade`.

2️⃣ Set up a palette. Recommended save directory is `resource/palette_files`.
Check out the [palette tutorial](../../palette_tut.md).

3️⃣ Execute `contour_shade.py`.
**Example**:
```commandline
python contour_shade.py --image_path ../../resource/imgs/tsunami.jpg --resize_factor 4 --contrast_factor 16 --resize_method bilinear
```

**Parameters**

| argument           | help                                                                                                                      |
|--------------------|---------------------------------------------------------------------------------------------------------------------------|
| --image_path       | The path of the image.                                                                                                    |
| --save_path        | The directory where the result image will be saved to.                                                                    |
| --resize_method    | The image resize method. Check below for available options.                                                               |
| --resize_factor    | The resize factor of the new image.                                                                                       |
| --contrast_factor  | The contrast factor based on the original image.                                                                          |
| --sigma_s          | The value of color smoothing in area.                                                                                     |
| --sigma_r          | The value of color smoothing in edges.                                                                                    |
| --thresholds_gamma | Controls the shading (gradient) levels. Higher value makes the algorithm emphasis the bright pixels. (Better granularity) |
| --palette_path     | Use a palette.                                                                                                            |
| --max_workers      | The maximum number of multithread workers.                                                                                |
| --invert_color     | If included, invert the color of the result image.                                                                        |
| --sigmaX           | The standard deviation of GaussianBlur kernel in the horizontal direction.                                                |
| --ksize            | The kernel size of contour function.                                                                                      |
| --gx               | The image gradient in the horizontal edges.                                                                               |
| --gy               | The image gradient in the vertical edges.                                                                                 |
| --color_option     | The option to color the image. Check below for available options.                                                         |
| --save_ascii       | If included, the characters will be saved to a file.                                                                      |
| --save_ascii_path  | The path to save the characters. Check out the 'ascii_output' folder for the results.                                     |

**resize_method**

| code             | help                                          |
|------------------|-----------------------------------------------|
| nearest neighbor | Resize image with nearest neighbor algorithm. |
| bilinear         | Resize image with bilinear algorithm.         |

**color_option**

| code     | help                                                           |
|----------|----------------------------------------------------------------|
| original | Color the ASCII art with the (resized) original image's color. |

An example of ascii art image:

<p align="center">
    <img src="../../resource/readme_imgs/eg_tsunami.png" width="400">
</p>

---

🖼️ Also check out the [gallery](./gallery.md) for more examples!

---

⭐ Image Credit: tsunami (Wikipedia)

<p align="center">
    <img src="../../resource/imgs/tsunami.jpg" width="400">
</p>
