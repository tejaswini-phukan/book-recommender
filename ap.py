import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("AI-Based Book Recommendation System")
st.write("Rate your preferences and get a book suggestion instantly!")

# Load dataset
data = pd.read_csv("book_data.csv")
X = data.drop("Book", axis=1)
y = data["Book"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# User inputs
fiction = st.slider("Fiction", 0, 10, 5)
mystery = st.slider("Mystery", 0, 10, 5)
scifi = st.slider("Sci-Fi", 0, 10, 5)
romance = st.slider("Romance", 0, 10, 5)
age = st.slider("Your Age", 10, 100, 25)
mood = st.slider("Mood Level", 0, 10, 5)

# Predict on button click
if st.button("Recommend Book"):
    user_input = [[fiction, mystery, scifi, romance, age, mood]]
    prediction = model.predict(user_input)[0]
    st.success(f"Recommended Book for You: **{prediction}**")
