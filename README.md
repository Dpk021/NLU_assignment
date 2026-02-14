# 🧠 Sports vs Politics Text Classification — ML From Scratch

A Machine Learning from Scratch project that builds a full NLP classification pipeline without using high-level ML model APIs.  
The system classifies documents into **Sports** or **Politics** using custom implementations of TF-IDF, Logistic Regression, Multinomial Naive Bayes, and Decision Tree.

---

## 📌 Project Overview

This project demonstrates an end-to-end text classification workflow:

1. Load and validate dataset  
2. Clean and tokenize text  
3. Build vocabulary and Bag-of-Words analysis  
4. Convert text into TF-IDF vectors  
5. Train multiple ML models from scratch  
6. Evaluate performance using classification metrics

The main goal is to understand the **internal working of ML algorithms** instead of using prebuilt libraries.

---

## ✨ Features

- Custom TF-IDF Vectorizer with smoothing and L2 normalization
- Logistic Regression using Batch Gradient Descent
- Multinomial Naive Bayes with Laplace smoothing
- Decision Tree using Gini impurity
- Custom Pipeline similar to sklearn.pipeline
- Logging-based dataset validation
- Performance evaluation using classification report

---

## 🧰 Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn (only for train-test split and metrics)
- Logging

---

## 📂 Project Structure

```
project/
│── sports_politics_dataset.csv
│── m25mac004_prob4.py
│── README.md
```

---

## ⚙️ Installation


Install required libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## ▶️ Usage

Run the program from terminal:

```bash
python m25mac004_prob4.py
```

The script will:

- Load dataset
- Perform preprocessing
- Train models from scratch
- Print classification reports for:

```
 Logistic Regression (Scratch)
 Multinomial Naive Bayes (Scratch)
 Decision Tree (Scratch)
```

---

## 🧹 Data Preprocessing

- Null value checking
- Lowercase normalization
- Tokenization
- Vocabulary creation
- TF-IDF feature extraction

---

## 🧮 Implemented Algorithms

### 🔵 Logistic Regression (From Scratch)
- Sigmoid activation
- Gradient Descent optimization
- Binary classification support
- One-vs-Rest multiclass strategy

### 🟢 Multinomial Naive Bayes (From Scratch)
- Log probability modeling
- Laplace smoothing
- Efficient probability computation

### 🌳 Decision Tree (From Scratch)
- Gini impurity splitting
- Feature sampling
- Recursive tree construction

---

## 🔁 Custom Pipeline

Pipeline flow:

```
Text → TFIDF → Model → Prediction
```

Provides a modular workflow similar to Scikit-learn pipelines.

---

## 📊 Evaluation

Models are evaluated using:

```
classification_report()
```

Metrics include:

- Precision
- Recall
- F1-Score
- Accuracy

---

## 🎯 Learning Objectives

- Understand ML algorithms internally
- Implement NLP feature engineering manually
- Build reusable ML pipelines from scratch
- Strengthen mathematical intuition behind models

---



