# 🧠 Autiscan - Autism Prediction Using Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-Hardik0385-blue?style=flat&logo=github)](https://github.com/Hardik0385/Autiscan)
[![Python](https://img.shields.io/badge/Python-3.x-yellow?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

A machine learning pipeline to predict Autism Spectrum Disorder (ASD) using the AQ-10 (Autism Quotient) screening questionnaire dataset.

## 📋 Overview

This project implements multiple classification algorithms to predict autism traits based on behavioral screening data. The AQ-10 is a widely-used 10-question screening tool designed to identify individuals who may benefit from a full diagnostic assessment.

## 🚀 Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables
- **Multiple ML Models**: 
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Feature Importance Analysis**: Identifies key predictors of ASD
- **Data Visualizations**: Interactive charts using Matplotlib & Seaborn
  - Class distribution (Pie & Bar charts)
  - Confusion matrices for all models
  - Model performance comparison
  - Feature importance bar chart

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Random Forest 🏆 | 84.38% |
| KNN | 83.12% |
| Logistic Regression | 82.50% |
| SVM | 79.38% |

## 📁 Dataset

The dataset contains 800+ samples with the following features:
- **A1-A10 Scores**: Responses to 10 behavioral screening questions (0 or 1)
- **Demographics**: Age, gender, ethnicity
- **Medical History**: Jaundice at birth, family history of autism
- **Target**: Class/ASD (0 = No ASD, 1 = ASD)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Hardik0385/Autiscan.git
cd Autiscan

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Make sure virtual environment is activated
python autism_prediction.py
```

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

> 💡 **Tip**: Install all dependencies at once using `pip install -r requirements.txt`

## 🔑 Key Findings

**Top Predictive Features:**
1. `result` - Overall AQ-10 screening score (16.4%)
2. `age` - Patient age (14.5%)
3. `A6_Score` - Question 6 response (12.7%)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git fork https://github.com/Hardik0385/Autiscan.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit
   ```bash
   git commit -m "Add: your feature description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub

### Contribution Ideas

- 🧪 Add more ML models (XGBoost, Neural Networks)
- 📊 Improve data visualization
- 🔧 Hyperparameter tuning
- 📝 Add unit tests
- 📖 Improve documentation

## 👥 Collaboration

Interested in collaborating on this project? I'd love to hear from you!

- **Email**: [Open an issue](https://github.com/Hardik0385/Autiscan/issues) on GitHub
- **LinkedIn**: Connect with me for discussions
- **GitHub Issues**: Report bugs or suggest features

### Areas for Collaboration

- Deep learning model implementation
- Web application development (Flask/Streamlit)
- Mobile app integration
- Research and paper writing

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⭐ Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

## 📚 References

- [Autism Spectrum Quotient (AQ)](https://www.autismresearchcentre.com/arc_tests/)
