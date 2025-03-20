# NLP Classification Models

This repository contains various Jupyter notebooks demonstrating the application of different machine learning classification algorithms on a Natural Language Processing (NLP) task. Each notebook focuses on a specific algorithm, walking through data preprocessing, model training, and evaluation.

## Repository Contents

- **NLP_LogisticRegression.ipynb**  
  Implements a Logistic Regression model on NLP data.

- **NLP_KNN.ipynb**  
  Implements a K-Nearest Neighbors classifier.

- **NLP_SVM.ipynb**  
  Implements a (linear) Support Vector Machine classifier.

- **NLP_KernelSVM.ipynb**  
  Implements a Kernel SVM classifier (e.g., RBF kernel).

- **NLP_DecisionTree.ipynb**  
  Implements a Decision Tree classifier.

- **NLP_NaiveBayes.ipynb**  
  Implements a Naive Bayes classifier.

- **NLP_RandomForest.ipynb**  
  Implements a Random Forest classifier.

## Overview

In these notebooks, we explore how different classification algorithms perform on a text classification problem. The general workflow includes:

1. **Data Loading and Exploration**  
   - Reading and inspecting the dataset.  
   - Understanding the distribution of classes.

2. **Data Preprocessing**  
   - Text cleaning (removing punctuation, lowercasing, etc.).  
   - Tokenization and vectorization (CountVectorizer).  
   - Splitting the dataset into training and testing sets.

3. **Model Training**  
   - Training each model on the preprocessed text data.  
   - Hyperparameter tuning (where applicable).

4. **Evaluation**  
   - Assessing model performance using metrics like accuracy.  
   - Comparing results across different algorithms.

## Performance Metrics

Below are the accuracy scores obtained from each model:

| Model                | Accuracy |
|----------------------|---------:|
| Logistic Regression  | 0.775    |
| K-Nearest Neighbors  | 0.66     |
| SVM (Linear)         | 0.785    |
| Kernel SVM           | 0.77     |
| Decision Tree        | 0.76     |
| Naive Bayes          | 0.73     |
| Random Forest        | 0.73     |

From these results, **SVM (Linear)** shows the highest accuracy at **0.785**, closely followed by **Logistic Regression** at **0.775** and **Kernel SVM** at **0.77**
