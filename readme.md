# 🎨 Polygon Colorization using Conditional UNet

This project implements a **Conditional UNet** model trained from scratch to colorize polygon shapes based on a given color name (e.g., `"blue"`, `"red"`). The model learns from paired examples of grayscale polygon images and their corresponding colored outputs, conditioned on color names.

---

## 🧠 Problem Statement

**Input:**
- A grayscale image of a polygon (e.g., triangle, square, octagon).
- A text color name (e.g., "red").

**Output:**
- An RGB image of the polygon filled with the specified color.

The challenge is to conditionally generate realistic polygon images filled with the specified color using deep learning techniques.

---

## 📁 Dataset

The dataset consists of:
dataset/
├── training/
│ ├── inputs/ # Input grayscale polygon images
│ ├── outputs/ # Corresponding colored output images
│ └── data.json # Mapping from input polygon & color → output image
├── validation/
│ ├── inputs/
│ ├── outputs/
│ └── data.json

## 🏗️ Model Architecture

A modified **UNet** model is used. It accepts the polygon image and a color condition via embedding. The embedding is broadcasted and concatenated with the image channels before passing through the UNet.

**Conditioning Strategy:**
- A learned `nn.Embedding` is used to represent color names.
- The embedding is spatially expanded and concatenated with the input image.

**Output:**
- A 3-channel RGB image generated using a final `sigmoid` layer.

---

## 🛠️ Implementation Details

| Component       | Value                         |
|----------------|-------------------------------|
| Model          | Conditional UNet              |
| Input Image    | RGB, 128x128                  |
| Output Image   | RGB, 128x128                  |
| Loss Function  | L1 Loss                       |
| Optimizer      | Adam (lr = 1e-3)              |
| Epochs         | 100                           |
| Batch Size     | 16                            |
| Conditioning   | Color Embedding (dim=16)      |
| Logging        | Weights & Biases (wandb)      |

---

## 📊 Training Logs

We track:
- **Training and Validation Losses**
- **Sample output visualizations**

Check the training progress on [Weights & Biases] (https://wandb.ai/010sachinmishra-international-institute-of-information-t/polygon-colorizer/runs/fw8x478p)

(https://wandb.ai/010sachinmishra-international-institute-of-information-t/polygon-colorizer)

https://wandb.ai/010sachinmishra-international-institute-of-information-t/polygon-colorizer?nw=nwuser010sachinmishra
<img width="641" height="346" alt="image" src="https://github.com/user-attachments/assets/5c120909-821f-4f77-bef6-bcfc52e936be" />
<img width="642" height="325" alt="image" src="https://github.com/user-attachments/assets/01e25a9f-6e22-4ef5-b7e2-52e51123cf52" />

## 🧪 Inference Demo

After training, run the inference notebook:
```python
# Load model and predict
input_image = 'octagon.png'
color_name = 'blue'

# Output will show side-by-side comparison
