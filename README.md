# 🎬 Sentiment Analyzer — NLP Project

An NLP project that classifies movie reviews as Positive or Negative
using TF-IDF vectorization and Logistic Regression.
Built with Python and deployed as a live web app using Streamlit.

## 🔗 Live Demo
👉(https://huggingface.co/spaces/codewithishu/sentiscope)

## 🏆 Model Performance
- **Accuracy: 83.50%**
- Trained on 2000 real IMDB movie reviews
- Tested on 400 unseen reviews

## 🔍 What I Did
- Loaded NLTK movie reviews dataset (2000 reviews)
- Cleaned text — lowercasing, punctuation removal, stopword removal
- Converted text to numbers using TF-IDF (5000 features)
- Trained Logistic Regression classifier
- Evaluated with accuracy, precision, recall, F1-score
- Deployed as interactive web app using Streamlit

## 🎯 Live Demo Results
| Review | Prediction | Confidence |
|---|---|---|
| "This movie was absolutely fantastic!" | POSITIVE 😊 | 94%+ |
| "Terrible film. Waste of time." | NEGATIVE 😞 | 90%+ |
| "One of the best movies ever seen!" | POSITIVE 😊 | 86.0% |
| "Horrible plot, bad acting..." | NEGATIVE 😞 | 81.9% |

## 🎯 Real Time Prediction Demo
After running, the model enters interactive mode:
```
🎯 TRY IT YOURSELF — REAL TIME PREDICTOR
==================================================
Type any movie review and I'll predict sentiment!
Type 'quit' to exit

Enter your review: The movie was absolutely brilliant!
✅ Prediction : POSITIVE 😊
📊 Confidence : 91.3%

Enter your review: Worst film ever, complete waste of time
❌ Prediction : NEGATIVE 😞
📊 Confidence : 88.7%
```

## 📈 Visualizations
| Chart | Description |
|---|---|
| Confusion Matrix | True vs predicted labels heatmap |
| Sentiment Distribution | Dataset balance check |
| Top 20 Words | Most important positive sentiment words |

## 🛠️ Tools & Libraries
- Python 3
- NLTK — text data & stopwords
- Scikit-learn — TF-IDF & Logistic Regression
- Pandas & NumPy — data handling
- Matplotlib & Seaborn — visualizations
- Streamlit — web app deployment

## 🚀 How to Run Locally
```bash
git clone https://github.com/codewithishu/sentiment-analyzer
cd sentiment-analyzer
pip install nltk scikit-learn pandas numpy matplotlib seaborn streamlit
python setup.py
streamlit run app.py
```

## 💡 Key Finding
Words like "excellent", "wonderful", "brilliant" are strongest
indicators of positive sentiment while "waste", "boring",
"awful" strongly indicate negative sentiment.

## 👨‍💻 About
Built as part of my Data Science portfolio.
