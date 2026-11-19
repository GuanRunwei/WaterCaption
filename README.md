# WaterCaption: A Fine-Grained Image Captioning Dataset for Inland Waterways

## 📘 Overview

**WaterCaption** is a large-scale, fine-grained image captioning dataset specifically designed for **inland waterway scenes** based on **Unmanned Surface Vehicles (USVs)**. WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size.


---

## 📂 Dataset Structure
### Download link
[baidu disk](https://pan.baidu.com/s/1U6V_U0KfOFSYOfnK5ZuC-w?pwd=bjcb)
password: bjcb
```
WaterCaption/
├── train/
│ ├── captions/
│ ├── images/
├── test/
└── val/
```


Each image is annotated with multiple captions describing scene elements such as ships, buoys, bridges, and water conditions.

---

## 📊 Statistics

- **Images**: 20.2k
- **Average captions per image**: 1
- **Scene types**: Rivers, ports, bridges, canals, ……
- **Annotation format**: mllm-type (e.g. LLaVA, InternVL, ……)

---

## 🚀 Baseline Models

We release several benchmark models:

### 🧠 InternVL + Baseline


### ⚡ MobileVLM + Baseline

---

## 🔧 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/GuanRunwei/WaterCaption.git
cd WaterCaption
conda create -n watercatpion python=3.10
conda activate watercaption

pip install -r requirements.txt
pip install -r requirements_mobileclip.txt
```

## 📘 Citation
```
@article{guan2025yu,
  title={Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding},
  author={Guan, Runwei and Ouyang, Ningwei and Xu, Tianhao and Liang, Shaofeng and Dai, Wei and Sun, Yafeng and Gao, Shang and Lai, Songning and Yao, Shanliang and Hu, Xuming and others},
  journal={arXiv preprint arXiv:2506.19288},
  year={2025}
}
```


