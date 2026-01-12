# 🎨 Real ASCII

> A fast, CLI-first tool for turning images into high-fidelity ASCII art using advanced shading and edge analysis.

<p align="center">
  <img src="resource/readme_imgs/main_logo.png" width="100%" alt="Project banner">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/kolyn090/real-ascii?style=flat-square">
  <img src="https://img.shields.io/github/license/kolyn090/real-ascii?style=flat-square">
</p>

---

## ✨ Features

- 🔍️ High-resolution ASCII image conversion
- 📏 Supports variable-width (non-monospace) characters
- 🎨 Full color rendering support
- 🧵 Preserves character anti-aliasing for smoother output
- 🧩 Modular and easy to extend
- 📦 Lightweight with minimal dependencies
- 🖼️ Supports PNG / JPG
- 💻 CLI + Library usage

---

## 🖼️ Gallery (Monospace)

> Example results generated using this library.

### Original

<p align="center">
  <img src="resource/imgs/flamingo.jpg" width="200">
</p>

### Filters Preview

| Edge Trace                                                            | Depth Shade                                                           | Contour Shade                                                           |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------|
| <img src="resource/readme_imgs/main_flamingo_trace.png" width="100%"> | <img src="resource/readme_imgs/main_flamingo_shade.png" width="100%"> | <img src="resource/readme_imgs/main_flamingo_contour.png" width="100%"> |

---

## 🖼️ Gallery (Non-Monospace)

> Example results generated using this library.

### Original

<p align="center">
  <img src="resource/imgs/sunflower.jpg" width="200">
</p>

### Filters Preview

| Edge Trace                                                             | Depth Shade                                                            | Contour Shade                                                            |
|------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------|
| <img src="resource/readme_imgs/main_sunflower_trace.png" width="100%"> | <img src="resource/readme_imgs/main_sunflower_shade.png" width="100%"> | <img src="resource/readme_imgs/main_sunflower_contour.png" width="100%"> |

---

## 🚀 Installation

### Using Git

```bash
git clone https://github.com/Kolyn090/real-ascii.git
cd real-ascii
pip install -r requirements.txt
```

---

## 🧭 Detail

[**Edge Trace ASCII Filter**](src/edge_trace/README.md)

Edge Detection + ASCII character matching

<p align="center">
    <img src="./resource/readme_imgs/main_girl.png" width="600">
</p>

<p align="center">
    <img src="./resource/readme_imgs/main_tsunami.png" width="600">
</p>

🖼️ Go to [gallery](src/edge_trace/gallery.md) to see more examples!

---

[**Depth Shade ASCII Filter**](src/depth_shade/README.md)

Each shading (gradient) level has its own set of characters. Just need to 
change one value (thresholds_gamma) to make the algorithm automatically 
distinguish the gradient level for your!

<p align="center">
    <img src="./resource/readme_imgs/main_monalisa.png" width="600">
</p>

🖼️ Go to [gallery](src/depth_shade/gallery.md) to see more examples!

---

[**Contour Shade ASCII Filter**](src/contour_shade/README.md)

Shade around the edges.

<p align="center">
    <img src="./resource/readme_imgs/main_tsunami2.png" width="600">
</p>

🖼️ Go to [gallery](src/contour_shade/gallery.md) to see more examples!

---

⭐ Image Credit: 
* girl with pearl earring by Johannes Vermeer (Wikipedia)
* tsunami by hokusai (Wikipedia)
* monalisa by Leonardo da Vinci (Wikipedia)
* [flamingo](https://pin.it/N40Wiy6zx)
* [sunflower](https://pin.it/tcKqTUF4G)
