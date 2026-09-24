# Credit Card Fraud Detection Using Hybrid Autoencoder and Random Forest

A hybrid machine learning and deep learning approach for detecting fraudulent credit card transactions using an **Autoencoder (AE)** for anomaly detection and latent feature extraction, combined with a **Random Forest (RF)** classifier.

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a small portion of the overall transaction data. This creates a highly imbalanced classification problem.

This project proposes a **Hybrid Autoencoder + Random Forest (AE+RF)** approach. The Autoencoder learns the underlying patterns of normal transactions and extracts latent representations along with reconstruction errors. These features are combined with the original transaction features and provided to a Random Forest classifier for final fraud classification.

### Key Techniques

- Autoencoder for anomaly detection
- Latent feature extraction
- Reconstruction error analysis
- Random Forest classification
- SMOTE for class imbalance
- PCA and feature scaling
- t-SNE visualization
- ROC-AUC and Precision-Recall evaluation

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Learn hidden representations using an Autoencoder.
- Identify anomalous transactions using reconstruction error.
- Combine Autoencoder features with original transaction features.
- Use Random Forest for final fraud classification.
- Handle class imbalance using SMOTE.
- Evaluate the model using multiple classification metrics.

## 🧠 Methodology

The overall workflow of the project is:

```text
Credit Card Dataset
        ↓
Data Preprocessing
        ↓
Feature Scaling / PCA
        ↓
Train / Validation / Test Split
        ↓
Autoencoder
        ↓
Latent Features + Reconstruction Error
        ↓
Combine with Original Features
        ↓
SMOTE
        ↓
Random Forest Classifier
        ↓
Fraud / Legitimate Prediction
```

The Autoencoder is trained to learn the underlying structure of normal transactions. Its latent representations and reconstruction errors are then combined with the original features and passed to the Random Forest classifier.

## 🔍 Autoencoder

The Autoencoder learns a compressed representation of the transaction data.

```text
Input Features
      ↓
   Encoder
      ↓
Latent Representation
      ↓
   Decoder
      ↓
Reconstructed Input
```

The reconstruction error is used as an additional feature for identifying potentially anomalous transactions.

## 🌲 Random Forest

The Random Forest classifier uses a hybrid feature representation consisting of:

- Original transaction features
- Autoencoder latent features
- Reconstruction error

The Random Forest then predicts whether a transaction is:

```text
0 → Legitimate
1 → Fraudulent
```

## ⚖️ Handling Class Imbalance

Credit card fraud datasets are highly imbalanced because legitimate transactions significantly outnumber fraudulent transactions.

To address this problem, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied during model training to increase the representation of the minority fraud class.

## 📊 Model Evaluation

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- t-SNE visualization
- Autoencoder reconstruction error
- Autoencoder training loss
- Random Forest feature importance

## 📈 Results

The hybrid AE+RF approach demonstrated very high classification performance on the evaluated test data, with precision, recall and ROC-AUC reported close to 1.0.

The project also uses:

- ROC curves
- Precision-Recall curves
- t-SNE plots
- Feature importance
- Training loss curves
- Confusion matrices

to analyze model performance and the learned feature representations.

## 📂 Dataset

The original dataset is **not included directly in this GitHub repository because it exceeds GitHub's 100 MB file-size limit.**

### Download Dataset

👉 **[Download the Dataset from Google Drive](https://drive.google.com/file/d/1KqRgumX2KomT_aTu11LqKO1CGlxA1aoj/view?usp=sharing)**

After downloading the dataset, place the CSV file inside the `data` folder:

```text
data/
└── dataset.csv
```

> **Note:** The `dataset.csv` file currently present in this repository is only a placeholder containing the download information. Replace it with the downloaded dataset before running the project.

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── dataset.csv
│   └── README.md
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── autoencoder.py
│   ├── random_forest.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── tsne.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

> **Note:** Update the file names above if your actual repository uses different names.

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest
- SMOTE
- PCA
- t-SNE

### Deep Learning

- TensorFlow
- Keras
- Autoencoder

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Google Colab
- T4 GPU

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate to the project directory:

```bash
cd YOUR_REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

### Step 1 — Download the Dataset

Download the original CSV from the Google Drive link provided above.

### Step 2 — Add the Dataset

Place the downloaded CSV inside:

```text
data/dataset.csv
```

### Step 3 — Run the Notebook

Open the project notebook and run the cells sequentially.

The workflow includes:

1. Loading the dataset
2. Data preprocessing
3. Feature scaling
4. Dataset splitting
5. Autoencoder training
6. Latent feature extraction
7. Reconstruction error calculation
8. SMOTE oversampling
9. Random Forest training
10. Prediction
11. Model evaluation
12. Visualization

## 🔬 System Architecture

The system consists of three major stages:

### Input Layer

Receives preprocessed and scaled transaction features.

### Autoencoder Layer

Learns latent representations and calculates reconstruction errors for anomaly detection.

### Random Forest Layer

Uses the original features, latent features and reconstruction error to classify transactions as fraudulent or legitimate.

## ⚠️ Limitations

- Autoencoder performance can be sensitive to hyperparameter selection.
- Model performance depends on the quality of the input data.
- Highly imbalanced datasets may still result in false negatives.
- Real-time deployment on large transaction streams may require significant computational resources.
- The internal representations of the hybrid model can be difficult to interpret.

## 🔮 Future Work

Possible future improvements include:

- Explainable AI (XAI)
- Deep ensemble learning
- Graph-based neural networks
- Real-time adaptive learning
- Cloud-based fraud detection
- Continuous model retraining
- Secure federated learning

## 📄 Research Publication

**A Hybrid Approach Using Autoencoder and Random Forest for Credit Card Fraud Detection**

**Authors:**  
Suraj Dehury  
Leena Das

**Institution:**  
Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar, India

**Conference:**  
2026 International Conference on Emerging Systems and Intelligent Computing (ESIC)

## 👨‍💻 Author

### Suraj Dehury

M.Tech – Computer Science & Engineering  
Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar

### Research Interests

- Machine Learning
- Data Analytics
- Deep Learning
- Fraud Detection
- Anomaly Detection
- Financial Security

## 📜 License

This project is intended for academic and research purposes.

Please refer to the original dataset's licensing and usage terms before redistributing the dataset.
