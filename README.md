# 🧠 Driver Drowsiness Detection using YOLOv5 and LSTM

Real-time drowsiness detection system that uses **YOLOv5** for face/eye detection and **LSTM** for fatigue classification based on temporal patterns.
This AI-based solution aims to enhance driver safety by detecting early signs of drowsiness.

---
![Adsız tasarım](https://github.com/user-attachments/assets/ad0bd633-e52d-400c-8741-0abcce5fc6be)
---

## 🚀 Features

* 🎥 Real-time webcam integration
* 👁️ Eye and face detection using YOLOv5
* ⏱️ Temporal sequence modeling with LSTM
* 🧠 Custom labeled dataset (awake/drowsy)
* 📊 Performance monitoring in Jupyter
* 🔧 Modular design for easy training and customization

---

## 🧠 How It Works

1. YOLOv5 model detects **face and eyes** from webcam feed.
2. Eye regions are cropped and passed to a trained **LSTM classifier**.
3. LSTM analyzes a **sequence** of frames to determine the state:

   * 🟢 `Awake`
   * 🔴 `Drowsy`

![part1](https://github.com/user-attachments/assets/ea55679a-a451-4f83-9bfb-43bbf28c8d61) 

![part2](https://github.com/user-attachments/assets/26595db1-c919-4164-bc98-a23dae8c4f7f)


> The system reacts instantly and overlays the prediction label with bounding boxes on the live feed.

---

## 📈 Model Evaluation Results

### 🔹 Label Distribution Heatmap



Labels are well-distributed in terms of position and size. Balanced annotation increases detection accuracy.

---

### 🔹 Precision vs Confidence

Awake class achieves high precision at lower confidence thresholds, while drowsy class stabilizes later.

---

### 🔹 Precision-Recall Curve

<p align="center">
  <img src="images/PR_curve.png" width="600">
</p>

* Awake: 0.948
* Drowsy: 0.861
* mAP\@0.5: 0.905

Indicates a reliable and balanced detection model.

---

### 🔹 Recall vs Confidence



High recall even at low confidence thresholds suggests effective detection capability.

---

### 🔹 Training Metrics

![P_curve](https://github.com/user-attachments/assets/e285202a-824f-41a5-bd9e-e4a8434b60f2)
![PR_curve](https://github.com/user-attachments/assets/844be9c7-08cd-464f-a132-023de5471800)
![R_curve](https://github.com/user-attachments/assets/271bc357-d3ae-48b7-a4be-639da5977d40)


Loss values decrease steadily and performance metrics increase over epochs, showing good training convergence.

---

### 🔹 Training Batches
![confusion_matrix](https://github.com/user-attachments/assets/72613b1e-68cf-4147-96b4-dfb109edd661)
![results](https://github.com/user-attachments/assets/3cc858f2-a0e6-458c-bb9b-afce5f363719)
![labels_correlogram](https://github.com/user-attachments/assets/2d9e285c-30c8-4c33-ba68-07391da69ca4)
![labels](https://github.com/user-attachments/assets/bc142546-3653-4073-92e6-31e7a9555756)
![F1_curve](https://github.com/user-attachments/assets/e205587a-a1fd-4dc0-860d-99ba77a1c428)


Sample training images with labels. Variations in expression, angle, and lighting enhance model generalization.

---

### 🔹 Validation Results

![train_batch0](https://github.com/user-attachments/assets/e5ecfdab-1da7-40cf-bcb1-d1b25f781e7f)
![train_batc![train_batch2](https://github.com/user-attachments/assets/32b72e46-c901-41ad-9ad6-b406f8606c62)
h1](https://github.com/user-attachments/assets/f7c48738-0ccf-465d-a5cd-f63b0c243145)
![val_batch0_labels](https://github.com/user-attachments/assets/d9252634-87ef-47ad-bdf2-9073c59d2da8)
![val_batch0_pred](https://github.com/user-attachments/assets/912da33a-8c72-444e-8bd8-edb68c97260e)



Predictions match the ground truth closely, validating model reliability.

---

## 📁 Project Structure

```
driver_drownise_detection_Program/
├── data/
├── datasets/
├── labelimg/
├── yolov5/
├── yolov5s.pt
├── train_lstm.py
├── main.py
├── Drowsiness Detection Tutorial.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/alinb/drowsiness-detection.git
cd drowsiness-detection
pip install -r requirements.txt
```

> YOLOv5 repo must be cloned under `yolov5/`:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

---

## 🦖 Training

### YOLOv5

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../data/data.yaml --weights yolov5s.pt
```

### LSTM

```bash
python train_lstm.py --data_dir datasets/ --epochs 50 --batch_size 32
```

---

## 📹 Run Real-Time Detection

```bash
python main.py
```

---

## 📊 Final Thoughts

The project demonstrates that combining object detection with temporal sequence modeling (YOLO + LSTM) can effectively solve real-world problems such as driver fatigue monitoring. Future improvements could include:

* Adding audio alerts
* Deploying on mobile devices with TFLite
* Expanding the dataset with night-time/low-light scenarios

---

## 📄 License

MIT © 2025 Ali Nebi Er

---

## 🤝 Acknowledgments

Built with support from OpenAI's GPT-4 and the YOLOv5 community.
