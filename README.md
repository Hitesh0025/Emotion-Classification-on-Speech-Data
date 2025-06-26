# 🎙️ Emotion Detection from Speech using Deep Learning

Welcome to my deep learning project where machines **learn to feel** — at least from sound! This is an end-to-end **Speech Emotion Recognition (SER)** system that processes raw `.wav` audio and identifies emotions like happiness, anger, sadness, and more.

---

## 🎯 Goal

To design a fully functional pipeline that:
- Extracts emotional cues from human voice using signal processing
- Classifies them using a custom deep learning model
- Offers an easy-to-use web interface built with Streamlit

Whether it’s a calm podcast or a dramatic song — this system picks up on the emotional tone.

---

## 💡 What It Does

- 🎵 Processes audio input (speech or song)
- 🎛 Extracts features like MFCC, ZCR, RMSE
- 🔄 Applies audio augmentation techniques
- 🧠 Runs inference using a 1D CNN
- 🌐 Serves real-time predictions via a Streamlit app

---

## 📦 Dataset

This project uses the **RAVDESS** dataset — a well-known benchmark for emotion classification tasks involving both speech and song recordings.

> 🎧 [Download Dataset](https://zenodo.org/records/1188976#.XCx-tc9KhQI)

### Dataset Stats

| Emotion   | Samples |
|-----------|---------|
| Calm      | 376     |
| Happy     | 376     |
| Sad       | 376     |
| Angry     | 376     |
| Fearful   | 376     |
| Disgust   | 192     |
| Surprised | 192     |
| Neutral   | 188     |

---

## ⚙️ How It Works

### 🔊 Feature Engineering
- **Sample Rate**: 22050 Hz
- **Windowing**: Frame = 2048, Hop = 512
- **Extracted Features**:
  - 13-dimensional MFCC
  - Zero Crossing Rate
  - Root Mean Square Energy  
- **Final Feature Vector**: Size = 2376

### 🔁 Audio Augmentation
To make the model robust to variations, we use:
- Time Stretching
- Pitch Shifting
- Background Noise
- Speed Variation
- Random Audio Shifts

### 🧱 Model Design

A **custom 1D CNN** was trained with:
- 5 stacked Conv1D blocks + MaxPooling + BatchNorm
- Dropout layers to reduce overfitting
- Dense layers topped with softmax for 8-class output

> 📊 Parameters: ~7.2 million

---

## 🧪 Training Configuration

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 40
- Batch Size: 64
- Dynamic Learning Rate: `ReduceLROnPlateau`

---

## 📊 Performance

| Metric        | Score     |
|---------------|-----------|
| Test Accuracy | 93.93%    |
| Macro F1      | 93.86%    |
| Micro F1      | 93.93%    |
| Weighted F1   | 93.94%    |

---

## 🌍 Web App Features (Streamlit)

The interactive app lets users:

- 📁 Upload one or many `.wav` files
- 🔊 Listen to each audio clip
- 📈 View prediction probabilities (bar/pie charts)
- 📉 See audio waveforms + distributions
- 📥 Export predictions to CSV

Live features:
- 🔄 Lazy model loading
- 🧠 Fast inference pipeline
- 🎯 Emotion distribution insights

---

## 🗂️ Project Layout

```

Emotion\_Classifier\_App/
├── app.py               # Streamlit interface
├── model/
│   ├── cnn\_model\_full.keras
│   ├── scaler.pkl
│   └── encoder.pkl
├── utils.py             # Feature extractors & helpers
├── notebooks/
│   └── training\_pipeline.ipynb
├── requirements.txt
└── assets/              # Sample audio clips (optional)

````

---

## 🧰 Tech Stack

- **ML/DL**: TensorFlow, Keras
- **Audio**: Librosa
- **UI/Visualization**: Streamlit, Plotly, Seaborn
- **Others**: Scikit-learn, NumPy, Pandas

---

## 💻 Run It Locally

### Step 1: Clone the repo
```bash
git clone https://github.com/your-username/emotion-classifier-app.git
cd emotion-classifier-app
````

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🙌 Credits

* RAVDESS Dataset by Ryerson University
* Open-source contributions from the deep learning community
* TensorFlow & Librosa teams

---

Let me know if you'd like to add:

* 🔧 Deployment instructions for Streamlit Cloud
* 📽️ Demo GIF/video
* 👥 Contributing or licensing guidelines

```


```
