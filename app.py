import nltk
import string
import streamlit as st
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope — AI Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0a0a0f;
    color: #f0f0f0;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a0a8ff 50%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.stat-chip {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: #aaa;
    font-family: 'DM Mono', monospace;
}

.stat-chip span {
    color: #a0a8ff;
    font-weight: 600;
}

.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace;
}

.result-box-pos {
    background: linear-gradient(135deg, #0d2018 0%, #0a1a10 100%);
    border: 1px solid #1a4d2e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.result-box-neg {
    background: linear-gradient(135deg, #1f0d0d 0%, #180a0a 100%);
    border: 1px solid #4d1a1a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.result-emoji {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}

.result-label-pos {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4ade80;
    font-family: 'Syne', sans-serif;
}

.result-label-neg {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f87171;
    font-family: 'Syne', sans-serif;
}

.confidence-text {
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.5rem;
    font-family: 'DM Mono', monospace;
}

.conf-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a0a8ff;
}

.example-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #444;
    margin: 1.5rem 0 0.75rem;
    font-family: 'DM Mono', monospace;
}

.footer-text {
    text-align: center;
    color: #333;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1a2a;
}

.footer-text a { color: #a0a8ff; text-decoration: none; }

div[data-testid="stTextArea"] textarea {
    background: #12121a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #a0a8ff !important;
    box-shadow: 0 0 0 2px rgba(160,168,255,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #a0a8ff, #ff6b9d) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #a0a8ff, #ff6b9d) !important;
    border-radius: 100px !important;
}

div[data-testid="stSpinner"] {
    color: #a0a8ff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Train Model ────────────────────────────────────────
@st.cache_resource
def train_model():
    import pandas as pd
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            documents.append((text, category))

    df = pd.DataFrame(documents, columns=['review', 'sentiment'])
    df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        return ' '.join(tokens)

    df['clean_review'] = df['review'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_review'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, tfidf, clean_text

# ── Hero Section ───────────────────────────────────────
st.markdown('<div class="hero-title">SentiScope</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// AI-powered sentiment analysis engine</div>', unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-chip">accuracy <span>83.50%</span></div>
    <div class="stat-chip">trained on <span>2,000 reviews</span></div>
    <div class="stat-chip">model <span>logistic regression</span></div>
    <div class="stat-chip">features <span>TF-IDF 5K</span></div>
</div>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────
with st.spinner("Initializing model..."):
    model, tfidf, clean_text = train_model()

# ── Input Section ──────────────────────────────────────
st.markdown('<div class="section-label">Input Review</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    placeholder="Paste any movie review here and hit Analyze...",
    height=140,
    label_visibility="collapsed"
)

analyze = st.button("⚡ Analyze Sentiment", use_container_width=True)

# ── Result ─────────────────────────────────────────────
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            clean = clean_text(user_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = max(prob) * 100
            pos_score = prob[1] * 100
            neg_score = prob[0] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-box-pos">
                <div class="result-emoji">😊</div>
                <div class="result-label-pos">POSITIVE SENTIMENT</div>
                <div class="confidence-text">confidence score — <span class="conf-value">{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-neg">
                <div class="result-emoji">😞</div>
                <div class="result-label-neg">NEGATIVE SENTIMENT</div>
                <div class="confidence-text">confidence score — <span class="conf-value">{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Prediction Breakdown</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Positive** — {pos_score:.1f}%")
            st.progress(int(pos_score))
        with col2:
            st.markdown(f"**Negative** — {neg_score:.1f}%")
            st.progress(int(neg_score))

# ── Examples ───────────────────────────────────────────
st.markdown('<div class="example-label">Quick Test Examples</div>', unsafe_allow_html=True)

examples = [
    "Absolutely brilliant film. Every scene was pure magic.",
    "Terrible movie. Boring plot, bad acting. Total waste.",
    "Great visuals but the story was weak and predictable.",
    "One of the best movies I have seen in years. Masterpiece!"
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        st.code(ex, language=None)

# ── Footer ─────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    built by <a href="https://github.com/codewithishu">Menka</a> ·
    <a href="https://github.com/codewithishu/sentiment-analyzer">view source</a> ·
    python + nltk + sklearn + streamlit
</div>
""", unsafe_allow_html=True)