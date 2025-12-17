# Shade ASCII Filter

The shade filter renders the image with shading.
Unlike in the trace filter, where palette is optional,
shade filter requires you to set up a palette. 

## 📖 Guide: Shade ASCII Art
1️⃣ `cd` to `src/shade`.

2️⃣ Set up a palette. Recommended save directory is `resource/palette_files`.

3️⃣ Execute `shade.py`.
**Example**:
```commandline
python shade.py --image_path ../../resource/imgs/monalisa.jpg --resize_factor 8
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

**resize_method**

| code             | help                                          |
|------------------|-----------------------------------------------|
| nearest neighbor | Resize image with nearest neighbor algorithm. |
| bilinear         | Resize image with bilinear algorithm.         |

An example of trace ascii art image (compressed):

<p align="center">
    <img src="../../resource/readme_imgs/shade_monalisa.png" width="400">
</p>

---

⭐ Image Credit: monalisa (Wikipedia)

<p align="center">
    <img src="../../resource/imgs/monalisa.jpg" width="400">
</p>
