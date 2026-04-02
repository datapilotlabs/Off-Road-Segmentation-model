# 🚀 Offroad Semantic Scene Segmentation

## 📥 Dataset
Download the dataset here:  
https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=HacktheNight  

---

## 📌 Overview
This project trains a semantic segmentation model on synthetic desert data and evaluates it on unseen images to test generalization.

---

## ⚙️ Setup

### Create Environment
```bash
conda create -n EDU python=3.10
conda activate EDU
```

### Install Dependencies
```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm pillow
```

---

## 📂 Project Structure
```
project/
│── train.py
│── test.py
│── dataset/
│── runs/
```

---

## 🧠 Train
```bash
python train.py
```

---

## 🧪 Test
```bash
python test.py
```

---

## 📊 Evaluation
- IoU (primary metric)  
- Loss metrics  
- Prediction outputs  

---

## ⚠️ Notes
- Do not use test images for training  
- Keep train/val/test data separate  

---

## 📦 Deliverables
- Trained model  
- Scripts (train.py, test.py)  
- Config files  
- Evaluation results  
