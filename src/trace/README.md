# Trace Filter

The trace image filter includes two steps. The first step generates
the contour images. You will choose one contour image that you want
to use for the next step. The second step generates the Trace ASCII 
Art. The following guides describes the process.

## 📖 Guide 1: Draw Image Contour
1️⃣ `cd` to `src/trace`.


2️⃣ Execute `contour.py`.
**Example**:
```commandline
python contour.py --image_path ../monalisa.jpg --canny1_min 0 --canny1_max 80 --canny1_step 20 --canny2_min 100 --canny2_max 180 --canny2_step 20
```

**Parameters**

| argument      | help                                                                       |
|---------------|----------------------------------------------------------------------------|
| --image_path  | The path of the image.                                                     |
| --save_folder | The folder where the contour images will be saved to.                      |
| --canny1_min  | The minimum value of threshold1 for Canny.                                 |
| --canny1_max  | The maximum value of threshold1 for Canny.                                 |
| --canny1_step | The step value of threshold1 for Canny.                                    |
| --canny2_min  | The minimum value of threshold2 for Canny.                                 |
| --canny2_max  | The maximum value of threshold2 for Canny.                                 |
| --canny2_step | The step value of threshold2 for Canny.                                    |
| --gb_size     | The kernel size of GaussianBlur.                                           |
| --gb_sigmaX   | The standard deviation of GaussianBlur kernel in the horizontal direction. |
| --kernel_size | The kernel size of contour function.                                       |
| --dilate_iter | The number of iterations of dilate.                                        |
| --erode_iter  | The number of iterations of erode.                                         |

---

## 📖 Guide 2: Trace ASCII Art


---

⭐ Image Credit: monalisa.jpa (Wikipedia)