# 💳 Credit Card Fraud Detection (ML Project)

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning models.
The goal is to correctly identify fraud cases while minimizing false alarms.

---

## 🎯 Objectives

* Build and compare classification models
* Handle class imbalance using `class_weight`
* Evaluate models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC Curve & AUC Score

---

## 📂 Dataset

* Dataset: `creditcard_2023.csv`
* Target variable: `Class`

  * `0` → Non-Fraud
  * `1` → Fraud

---

## ⚙️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 🧠 Models Used

1. Logistic Regression
2. Decision Tree Classifier

---

## 🔄 Workflow

1. Data Loading & Preprocessing
2. Feature Scaling (`StandardScaler` for Amount)
3. Train-Test Split (80/20 with stratification)
4. Model Training
5. Model Evaluation
6. ROC Curve & AUC Comparison

---

## 📊 Evaluation Metrics

* **Accuracy** → Overall correctness
* **Precision** → Correct fraud predictions
* **Recall** → Ability to detect fraud (very important)
* **F1-score** → Balance of precision & recall
* **AUC Score** → Model’s ability to separate classes

---

## 📈 Results

### Logistic Regression

* Accuracy: ~95%
* Strong and stable performance

### Decision Tree

* Accuracy: ~95%
* Tuned to avoid overfitting

---

## 📉 ROC Curve

* ROC Curve is used to evaluate performance across different thresholds
* AUC Score helps compare models:

  * Higher AUC = Better model

---

## 🧪 Key Learnings

* Accuracy alone is not enough for imbalanced datasets
* Recall is crucial in fraud detection
* Decision Trees can overfit if not controlled
* ROC-AUC is important for model comparison

---

## 🚀 Future Improvements

* Try Random Forest and XGBoost
* Perform hyperparameter tuning
* Use cross-validation
* Deploy as a web application

---

## 🙌 Conclusion

Both models performed well, achieving around 95% accuracy.
Logistic Regression provided stable results, while Decision Tree required tuning to prevent overfitting.
ROC-AUC helped in better understanding model performance beyond accuracy.

---

## 📎 How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
python your_script.py
```

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!
