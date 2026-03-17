import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import string

# ── Load IMDB Movie Reviews Dataset ───────────────────
print("⏳ Loading dataset...")
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        text = movie_reviews.raw(fileid)
        documents.append((text, category))

df = pd.DataFrame(documents, columns=['review', 'sentiment'])
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})
print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"   Positive reviews: {df['sentiment'].sum()}")
print(f"   Negative reviews: {(df['sentiment']==0).sum()}")

# ── Clean Text ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("\n⏳ Cleaning text data...")
df['clean_review'] = df['review'].apply(clean_text)
print("✅ Text cleaned!")

# ── TF-IDF Vectorization ───────────────────────────────
print("\n⏳ Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']
print(f"✅ Vectorized! Feature matrix shape: {X.shape}")

# ── Train Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train Model ────────────────────────────────────────
print("\n⏳ Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── Evaluate Model ─────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(f"\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ── Chart 1: Confusion Matrix ──────────────────────────
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('chart1_confusion_matrix.png', dpi=150)
plt.close()
print("✅ Chart 1 saved!")

# ── Chart 2: Sentiment Distribution ───────────────────
plt.figure(figsize=(6, 5))
df['sentiment'].map({1: 'Positive', 0: 'Negative'}).value_counts().plot(
    kind='bar', color=['#1D9E75', '#E24B4A'], edgecolor='none'
)
plt.title('Sentiment Distribution in Dataset', fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart2_sentiment_distribution.png', dpi=150)
plt.close()
print("✅ Chart 2 saved!")

# ── Chart 3: Top 20 Most Important Words ──────────────
plt.figure(figsize=(10, 7))
feature_names = tfidf.get_feature_names_out()
coef = model.coef_[0]
top_positive_idx = coef.argsort()[-20:]
top_words = [feature_names[i] for i in top_positive_idx]
top_scores = [coef[i] for i in top_positive_idx]
plt.barh(top_words, top_scores, color='#378ADD')
plt.title('Top 20 Words that Indicate Positive Sentiment', fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('chart3_top_words.png', dpi=150)
plt.close()
print("✅ Chart 3 saved!")

# ── Live Demo ──────────────────────────────────────────
print("\n" + "="*50)
print("🎬 SENTIMENT ANALYZER — LIVE DEMO")
print("="*50)

test_reviews = [
    "This movie was absolutely fantastic! I loved every moment of it.",
    "Terrible film. Waste of time and money. Very disappointing.",
    "The acting was great but the story was a bit slow.",
    "One of the best movies I have ever seen in my life!",
    "Horrible plot, bad acting, I fell asleep halfway through."
]

for review in test_reviews:
    clean = clean_text(review)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    confidence = max(prob) * 100
    print(f"\nReview    : {review[:60]}...")
    print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")

print("\n" + "="*50)
print(f"🏆 Final Model Accuracy: {acc*100:.2f}%")
print("="*50)
print("\n✅ All done! Check folder for 3 charts.")
# ── Interactive Real Time Predictor ───────────────────
print("\n" + "="*50)
print("🎯 TRY IT YOURSELF — REAL TIME PREDICTOR")
print("="*50)
print("Type any movie review and I'll predict sentiment!")
print("Type 'quit' to exit\n")

while True:
    user_review = input("Enter your review: ")
    
    if user_review.lower() == 'quit':
        print("👋 Thanks for using Sentiment Analyzer!")
        break
    
    if len(user_review.strip()) == 0:
        print("⚠️  Please enter a valid review!\n")
        continue
    
    clean = clean_text(user_review)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob) * 100
    
    if pred == 1:
        print(f"✅ Prediction : POSITIVE 😊")
    else:
        print(f"❌ Prediction : NEGATIVE 😞")
    
    print(f"📊 Confidence : {confidence:.1f}%\n")

