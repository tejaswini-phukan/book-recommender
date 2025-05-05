import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f9f7f7;
        background-image: url('https://images.unsplash.com/photo-1600488994401-d7bde26cbbf3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(4px);
        border-radius: 12px;
        padding: 20px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .book-box {
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    .header-img {
        width: 100%;
        max-height: 220px;
        object-fit: cover;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .divider-img {
        width: 80%;
        max-height: 100px;
        object-fit: cover;
        margin: 25px auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
data = pd.read_csv("book.csv")

# Header Image
st.markdown("""
<div style="text-align: center;">
    <img src="https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 
         class="header-img" 
         alt="Books and Coffee">
</div>
""", unsafe_allow_html=True)

st.title("📚 Personalized Book Recommendation")

# Dataset preview
with st.expander("📂 Preview Dataset"):
    st.dataframe(data.head())

# Model Training
X = data.drop(["Book", "image_url"], axis=1)
y = data["Book"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"✅ Model trained with Accuracy: **{accuracy:.2f}**")

# Decorative Divider
st.markdown("""
<div style="text-align: center;">
    <img src="https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 
         class="divider-img" 
         alt="Divider">
</div>
""", unsafe_allow_html=True)

# Preferences Input
st.header("🎯 Tell us about your preferences")

col1, col2 = st.columns(2)
with col1:
    fiction = st.slider("📖 Fiction", 1, 10, 5)
    mystery = st.slider("🕵️ Mystery", 1, 10, 5)
    scifi = st.slider("🤖 Sci-Fi", 1, 10, 5)

with col2:
    romance = st.slider("❤️ Romance", 1, 10, 5)
    age = st.number_input("🎂 Your Age", min_value=5, max_value=100, value=25)
    mood = st.slider("😊 Mood Level", 1, 10, 5)

# Predict Button
if st.button("📖 Recommend a Book"):
    user_input = [[fiction, mystery, scifi, romance, age, mood]]
    prediction = model.predict(user_input)[0]
    image_url = data.loc[data["Book"] == prediction, "image_url"].values[0]

    # Display Result
    st.markdown(f"""
    <div class="book-box">
        <h3>🎉 Recommended Book for You:</h3>
        <h2 style="color:#3b82f6;">📚 {prediction}</h2>
        <div style="text-align: center;">
            <img src="{image_url}" style="max-height: 300px; border-radius: 12px; margin-top: 10px;" />
        </div>
        <p style="margin-top: 20px; text-align: center;">✨ Enjoy your reading journey! ✨</p>
    </div>
    """, unsafe_allow_html=True)

    # Bottom Divider
    st.markdown("""
    <div style="text-align: center;">
        <img src="https://images.unsplash.com/photo-1510172951991-856a62a9e395?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" 
             class="divider-img" 
             alt="Happy reading">
    </div>
    """, unsafe_allow_html=True)

