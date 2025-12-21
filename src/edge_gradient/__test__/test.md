# Test 1

<p align="center">
    <img src="./imgs/test_1.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ❌        |
| Invert Image Color | ❌        |

```commandline
python edge_gradient.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--contrast_factor 1 ^
--thresholds_gamma 2
```

---

# Test 2

<p align="center">
    <img src="./imgs/test_2.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ✅        |
| Invert Image Color | ❌        |

```commandline
python edge_gradient.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--color_option original ^
--contrast_factor 1 ^
--thresholds_gamma 2
```

---

# Test 3

<p align="center">
    <img src="./imgs/test_3.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ✅        |
| Invert Image Color | ✅        |

```commandline
python edge_gradient.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--color_option original ^
--invert_color ^
--contrast_factor 1 ^
--thresholds_gamma 2
```

---

# Test 4

<p align="center">
    <img src="./imgs/test_4.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ❌        |
| Invert Image Color | ✅        |

```commandline
python edge_gradient.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--invert_color ^
--contrast_factor 1 ^
--thresholds_gamma 2
```
