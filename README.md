# 🎬 IMDB Movie Review Sentiment Analysis using Deep Learning (Optimized ANN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green)
![NLP](https://img.shields.io/badge/NLP-Sentiment--Analysis-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📘 Project Overview

This project applies **Deep Learning** techniques to perform **Sentiment Analysis** on the **IMDB Movie Review Dataset** using an **Artificial Neural Network (ANN)**.  
The model determines whether a movie review expresses a **positive** or **negative** sentiment.

It uses **TF-IDF** for text feature extraction and a carefully optimized neural network with:
- **Batch Normalization**
- **Leaky ReLU Activation**
- **Dropout Regularization**
- **L2 Weight Regularization**
- **Early Stopping**

The architecture is fine-tuned to achieve **robust accuracy (~87%)** with controlled overfitting.

---

## 🎯 Objective

To develop a **robust text classification model** that can accurately predict the sentiment polarity of IMDB movie reviews using optimized neural network architecture and advanced NLP preprocessing.

---

## 📂 Dataset Information

**Dataset Name:** [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
**Size:** 50,000 reviews (Balanced)  
- 25,000 for training  
- 25,000 for testing  

**Label Distribution:**
| Sentiment | Label | Count |
|------------|--------|--------|
| Positive | 1 | 25,000 |
| Negative | 0 | 25,000 |

Each review is a textual paragraph expressing a user’s opinion about a movie.

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Convert all text to lowercase  
- Remove HTML tags and punctuation  
- Keep only alphabetic characters  
- Encode sentiment labels (positive → 1, negative → 0)

### 2️⃣ Data Splitting
- 80% Training Data  
- 20% Testing Data  
- Random seed fixed at `42` for reproducibility

### 3️⃣ Feature Extraction
- **TF-IDF Vectorization**
  - `max_features = 15000`
  - `ngram_range = (1, 2)`
  - `stop_words = 'english'`

This converts textual reviews into a **15,000-dimensional numeric vector**, representing word importance.

### 4️⃣ Model Architecture

| Layer Type | Units | Activation | Regularization | Dropout | Notes |
|-------------|--------|-------------|----------------|----------|-------|
| Dense | 1024 | LeakyReLU(0.1) | L2(0.001) | 0.5 | Input Layer |
| Dense | 512 | LeakyReLU(0.1) | L2(0.001) | 0.4 | Hidden Layer |
| Dense | 256 | LeakyReLU(0.1) | L2(0.001) | 0.3 | Hidden Layer |
| Dense | 128 | LeakyReLU(0.1) | L2(0.001) | 0.3 | Hidden Layer |
| Dense | 64 | LeakyReLU(0.1) | L2(0.001) | 0.2 | Hidden Layer |
| Dense | 1 | Sigmoid | - | - | Output Layer |

Additional Enhancements:
- **BatchNormalization** after every layer for faster convergence  
- **LeakyReLU** prevents neuron death  
- **Dropout** ensures better generalization  
- **L2 Regularization** reduces weight explosion  

---

## 🧠 Model Compilation and Training

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 15 (Early Stopping) |
| Batch Size | 64 |
| Validation Split | 20% |
| Early Stopping | Patience = 2 |

**Early Stopping:** Stops training when `val_loss` no longer improves, ensuring the best weights are restored.

---

## 📊 Results and Analysis

### ✅ Final Evaluation Metrics

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.8712 |
| **Precision (Positive)** | 0.86 |
| **Recall (Positive)** | 0.88 |
| **F1-Score (Positive)** | 0.87 |
| **Precision (Negative)** | 0.88 |
| **Recall (Negative)** | 0.86 |
| **F1-Score (Negative)** | 0.87 |

---

### 🔍 Classification Report

           precision    recall  f1-score   support

       0       0.88      0.86      0.87      4961
       1       0.86      0.88      0.87      5039

accuracy                           0.87     10000


### 🧩 Confusion Matrix

|              | Predicted Negative | Predicted Positive |
|--------------|--------------------|--------------------|
| **Actual Negative (0)** | 4255 | 706 |
| **Actual Positive (1)** | 582  | 4457 |

**Interpretation:**
- 4255 reviews were correctly identified as negative  
- 4457 reviews were correctly identified as positive  
- The model misclassified only around **12.9%** of the total reviews  

---

### 📈 Training Behavior

- **Training Accuracy:** ↑ 93% → 94%  
- **Validation Accuracy:** Stabilized around 87%  
- **Loss Curve:** Validation loss converges early due to L2 regularization and early stopping  
- **No overfitting** observed — training and validation performance remain closely aligned  

---

## 🧰 Technologies Used

### 🔹 Programming Language
- **Python 3.8+**

### 🔹 Data Handling & Preprocessing
- **pandas** → Dataset manipulation  
- **numpy** → Array operations  
- **re (Regex)** → Text cleaning and pattern matching  

### 🔹 Natural Language Processing
- **scikit-learn**
  - `TfidfVectorizer` → Text vectorization  
  - `train_test_split` → Data partitioning  
  - `accuracy_score`, `classification_report`, `confusion_matrix` → Performance metrics  

### 🔹 Deep Learning Framework
- **TensorFlow / Keras**
  - `Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `LeakyReLU` → Neural network architecture  
  - `Adam Optimizer` → Adaptive gradient optimization  
  - `EarlyStopping` → Regularization and convergence control  
  - `l2` → Weight regularization  

### 🔹 Optional Visualization (Recommended)
- **Matplotlib** → For plotting accuracy/loss curves  
- **Seaborn** → For visualizing confusion matrix  

---
## 🚀 Future Improvements

| Enhancement | Description |
|--------------|-------------|
| 🔤 **Word Embeddings** | Replace TF-IDF with Word2Vec, GloVe, or FastText |
| 🧩 **Deep Architectures** | Use LSTM, GRU, or BiLSTM for sequential learning |
| 🌐 **Transfer Learning** | Integrate BERT or DistilBERT for contextual embeddings |
| 📈 **Visualization Dashboard** | Add training analytics via TensorBoard or Streamlit |
| 🧮 **Hyperparameter Optimization** | Tune learning rate, dropout, and regularization strength using Optuna |



---
👨‍💻 Author

Ali Khan
AI Engineer 
📧 alikhan132311@gmail.com


💡 Passionate about Deep Learning, NLP, and Model Optimization.

⭐ If you find this project helpful, please give it a star on GitHub!
