# Sentiment Analysis on IMDB Movie Reviews

## Project Overview
This project performs **Sentiment Analysis** on the **IMDB Dataset of 50K Movie Reviews** using **Natural Language Processing (NLP)** and **Machine Learning**. The dataset contains movie reviews labeled as **positive** or **negative** sentiments. The goal is to build a classification model that predicts the sentiment of a given review.

Dataset: [IMDB Dataset - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Features
- **Text Preprocessing**: Cleaning, Tokenization, Stopword Removal
- **Data Visualization**: Sentiment Distribution & Confusion Matrix
- **Word Embedding**: TF-IDF Vectorization
- **Machine Learning Model**: Naive Bayes Classifier
- **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix

## Workflow
### 1. Essential Imports
The necessary libraries such as Pandas, NumPy, Seaborn, Matplotlib, and Scikit-Learn are imported.

### 2. Data Loading
The IMDB dataset is loaded from a CSV file, containing two columns:
- `review`: The movie review text.
- `sentiment`: The sentiment label (positive/negative).

### 3. Text Preprocessing
The text is cleaned by:
- Converting to lowercase.
- Removing HTML tags.
- Removing punctuation and non-alphabetic characters.
- Removing stopwords.
- Tokenizing the cleaned text.

### 4. Data Visualization
A bar chart is generated to show the distribution of **positive** and **negative** reviews in the dataset.

### 5. Word Embedding
TF-IDF (**Term Frequency-Inverse Document Frequency**) is used to convert text into numerical features for training the machine learning model.

### 6. Splitting Data
The dataset is split into **80% training** and **20% testing** data.

### 7. Model Training
A **Naive Bayes Classifier** is trained using a pipeline that includes:
- **TF-IDF Vectorization**
- **Multinomial Naive Bayes Model**

### 8. Model Prediction & Evaluation
The trained model is used to make predictions on the test data. The following metrics are used to evaluate performance:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (Heatmap visualization)

## Results
The trained model achieves **high accuracy** in predicting the sentiment of movie reviews. The confusion matrix provides insights into the model’s classification performance.

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Download the Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project directory.

### 4. Run the Script
```bash
python sentiment_analysis.py
```

## Technologies Used
- **Python**
- **Pandas & NumPy** (Data Handling)
- **NLTK** (Text Processing)
- **Scikit-learn** (Machine Learning)
- **Seaborn & Matplotlib** (Visualization)

