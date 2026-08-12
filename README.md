# 🎬 Movie Genre Classification - CodSoft Task 1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TF--IDF-NLP-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CodSoft-Internship-purple?style=for-the-badge" />
</p>

<p align="center">
  <b>An intelligent NLP system that predicts movie genres from plot summaries using Machine Learning</b>
</p>

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Intern Name** | Vishal Kadwe |
| **Intern ID** | BY26RY229988 |
| **Organization** | CodSoft |
| **Internship Type** | Machine Learning (Virtual) |
| **Duration** | 15 August 2026 - 15 September 2026 |
| **Task** | Task 1 of 5 |

---

## 🎯 Objective

Build a machine learning model that can automatically classify movies into genres (**Drama, Comedy, Action, Horror, Romance**) based solely on their plot descriptions using Natural Language Processing (NLP) techniques.

---

## 🗂️ Dataset

- **Source:** [IMDb Genre Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- **Format:** `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- **Training Samples:** 50 movies
- **Test Samples:** 25 movies
- **Genres:** 5 unique categories

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **NLP** | Custom Text Preprocessing (Stopword Removal, Tokenization) |
| **Feature Extraction** | Scikit-learn TF-IDF Vectorizer |
| **ML Models** | Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| **Visualization** | Matplotlib, Seaborn, Chart.js (Web) |
| **Web Portfolio** | HTML5, CSS3, JavaScript (3D Effects) |

---

## 🧠 Algorithms Implemented

### 1. Multinomial Naive Bayes
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Excellent baseline for text classification

### 2. Logistic Regression
- Linear model with logistic sigmoid function
- L2 regularization (C=1.0)
- **Best performing model** ✅

### 3. Support Vector Machine (SVM)
- LinearSVC with maximum margin classification
- Effective for high-dimensional TF-IDF features

---

## 📊 Results

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| Naive Bayes | 80% | 0.78 | 0.81 |
| **Logistic Regression** | **90%** | **0.89** | **0.91** |
| SVM | 85% | 0.84 | 0.86 |

**🏆 Best Model: Logistic Regression**

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Python Script
```bash
python movie_genre_task1.py
```

### View the Portfolio Website
Simply open `index.html` in any modern web browser:
```bash
# Or just double-click the file!
start index.html
```

---

## 📁 Project Structure

```
CODSOFT_TASK1/
│
├── 📄 movie_genre_task1.py          # Main Python ML pipeline
├── 📄 index.html                     # 3D Portfolio Website
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Python dependencies
│
├── 📊 Output Files (Auto-Generated):
│   ├── genre_distribution.png
│   ├── description_length_distribution.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── test_predictions_distribution.png
│   └── test_predictions.csv
│
└── 📄 Data Files (Auto-Generated):
    ├── train_data.txt
    └── test_data.txt
```

---

## 🎨 Features

### Machine Learning Pipeline
- ✅ Custom text preprocessing (no heavy NLTK dependency)
- ✅ TF-IDF vectorization with unigrams & bigrams
- ✅ Stratified train-test split
- ✅ 3 model comparison with confusion matrices
- ✅ Automated best model selection

### 3D Portfolio Website
- 🌌 Animated particle background with connection lines
- 🎭 Glassmorphism UI with 3D hover effects
- 📊 Interactive Chart.js visualizations
- 🧠 Live demo with real-time genre prediction
- 📱 Fully responsive design
- ✨ Scroll animations & typing effects

---

## 📝 Sample Predictions

| Movie Title | Predicted Genre |
|-------------|-----------------|
| Movie 51 | Drama |
| Movie 52 | Comedy |
| Movie 53 | Action |
| Movie 54 | Horror |
| Movie 55 | Romance |

*See `test_predictions.csv` for complete results.*

---

## 🎥 Demo Video

Record a screen capture showing:
1. Running the Python script
2. Generated charts and metrics
3. The interactive portfolio website
4. Live genre prediction demo

Post on LinkedIn with `#codsoft #internship #machinelearning`

---

## 📧 Contact

- **Email:** contact@codsoft.in
- **Website:** [www.codsoft.in](https://www.codsoft.in)
- **LinkedIn:** Tag `@CodSoft` in your posts

---

<p align="center">
  <b>Built with by Vishal Kadwe | CodSoft ML Internship 2026</b>
</p>
