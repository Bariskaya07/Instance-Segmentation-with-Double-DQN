
## Double DQN ve Reinforcement Learning Yaklaşımıyla Instance Segmentation

Bu proje, standart **Mask R-CNN** mimarisine entegre edilen **Double Deep Q-Network (Double DQN)** yapısıyla güçlendirilmiş bir instance segmentation (nesne bazlı bölütleme) sistemi sunar. Amaç, piksel düzeyindeki segmentasyon kalitesini artırmak ve maske iyileştirme sürecini **IoU (Intersection over Union)** değerlerine dayalı **ödül sinyalleri** ile yönlendirilen ardışıl bir karar verme problemine dönüştürerek öğrenmeyi daha akıllı hale getirmektir.


## Reinforcement Learning Approach to Instance Segmentation Using Double DQN

This project presents a reinforcement learning-enhanced instance segmentation pipeline that integrates a **Double Deep Q-Network (Double DQN)** into a standard **Mask R-CNN** framework. The goal is to improve the pixel-level segmentation quality by treating mask refinement as a sequential decision-making problem, guided by **reward signals** based on **IoU (Intersection over Union)** improvements.
---
## 🧠 Motivation

Instance segmentation requires high-quality, fine-grained object masks — especially in complex scenes with occlusions and clutter. While **Mask R-CNN** is powerful, it relies heavily on supervised learning and large annotated datasets. This project introduces **reinforcement learning (RL)** to enhance mask quality with a **Double DQN** module that learns to refine predictions based on experience and feedback.

---

## 🧰 Key Contributions

1. **RL-enhanced Mask R-CNN**: Adds a Double DQN-based mask refinement policy to a pretrained Mask R-CNN.
2. **Reward-driven Optimization**: Rewards are based on IoU improvement, guiding the agent toward better mask alignment.
3. **No Additional Epochs Needed**: Significant performance boost is achieved without extending total training time.
4. **+57% AP boost**: Compared to a supervised-only Mask R-CNN baseline.
````markdown
## 📦 Project Structure

```bash
instance-segmentation-rl/
├── segmentation/               # Reinforcement-enhanced Mask R-CNN module
│   ├── class_names.py
│   ├── coco_eval.py
│   ├── coco_utils.py
│   ├── engine.py
│   ├── group_by_aspect_ratio.py
│   ├── infer_utils.py
│   ├── inference_image.py
│   ├── inference_video.py
│   ├── presets.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
├── results/                    # Result figures, metrics, and output videos
├── configs/                    # YAML or JSON configs for training
├── requirements.txt
└── README.md
````

---

## 🧪 Methodology

### 🧱 Base Architecture: Mask R-CNN

* Backbone: ResNet-50 FPN
* Dataset: COCO 2017 Instance Segmentation

### 🔁 RL Module: Double DQN

* State: Segmentation mask grid
* Action Space:

  * Dilation
  * Erosion
  * No-op
* Reward: IoU gain compared to previous mask
* Exploration: Epsilon-greedy strategy

### 🧮 Training Flow

* Epochs 1–3: Mask R-CNN trains with supervised loss
* Epochs 4–10: Double DQN takes over for iterative mask refinement
* Q-values updated with experience replay and target networks

---

## 📊 Results

| Metric                       | Baseline (Mask R-CNN) | RL-Enhanced (Ours) |
| ---------------------------- | --------------------- | ------------------ |
| AP@\[0.50:0.95]              | 0.171                 | **0.268** (+57%)   |
| AP\@0.50                     | 0.305                 | **0.444** (+46%)   |
| AP\@0.75                     | 0.169                 | **0.283** (+67%)   |
| AR@\[0.50:0.95] (maxDet=100) | 0.330                 | **0.338**          |

> 🧠 Our model achieved **faster convergence**, higher-quality masks, and better generalization without needing longer training schedules.

---

## 🔧 How to Use

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Mask R-CNN + Double DQN

```bash
python segmentation/train.py --epochs 10 --rl_start_epoch 4
```

### Run Inference on Images

```bash
python segmentation/inference_image.py --input input.jpg --output result.png
```

### Run Inference on Videos

```bash
python segmentation/inference_video.py --input input.mp4 --output result.mp4
```


## 📝 Citation (APA)

Kaya, B., Denkınalbant, A., Ahangari, M., Saidburkhan, A., & Arıcan, E. (2025). *Reinforcement Learning Approach to Instance Segmentation Using Double DQN*. Bahçeşehir University.

---

## 📜 License

MIT License — see [`LICENSE`](./LICENSE) for full details.

---

## 🙏 Acknowledgements

This research was supported by Bahçeşehir University. We also thank:

* **Dr. Erkut Arıcan** for mentoring and project supervision.
* Open-source contributors from **Detectron2**, **PyTorch**, and **COCO** dataset community.

---

```
```
