import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import euclidean_distances

# Page Config
st.set_page_config(page_title="Book Mate", page_icon="bookLogo.png", layout="centered")

# Custom CSS for UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }
    .main {
        background-image: url('https://images.unsplash.com/photo-1512820790803-83ca734da794?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        padding: 20px;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.1);
    }
    .stSidebar { 
        background-color: #f5f5f5; 
        padding: 20px; 
        border-radius: 0 16px 16px 0;
    }
    .stButton>button {
        background-color: #ef4444;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #dc2626;
        transform: translateY(-2px);
    }
    .book-box {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 25px;
        background-color: #ffffff;
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        margin-top: 30px;
        transition: all 0.3s ease;
    }
    .book-box:hover { 
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    .header-img {
        width: 100%;
        height: auto;
        max-height: 380px;
        object-fit: contain;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    ul { 
        padding-left: 20px;
        margin: 12px 0;
    }
    li {
        margin-bottom: 8px;
    }
    .pdf-link {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.6rem 1.2rem;
        background: #ef4444;
        color: white;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }
    .pdf-link:hover { 
        background: #dc2626;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(220, 38, 38, 0.4);
    }
    .book-description {
        background: #f8fafc;
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #ef4444;
        margin: 15px 0;
        line-height: 1.7;
        color: #4b5563;
        font-size: 15px;
    }
    .genre-badge {
        display: inline-block;
        padding: 4px 10px;
        background: #e2e8f0;
        border-radius: 20px;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .book-title {
        color: #1e293b;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f1f5f9;
    }
    .book-meta {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("book.csv")
    df['Book'] = df['Book'].str.strip()
    df = df[df['Language'] == 'English'].reset_index(drop=True)
    return df

df = load_data()

# PDF Links
pdf_links = {
    "The Great Gatsby": "https://www.gutenberg.org/cache/epub/64317/pg64317-images.html",
    "1984": "https://freeditorial.com/en/books/1984-by-george-orwell/readonline",
    "Pride and Prejudice": "https://freeditorial.com/en/books/pride-and-prejudice/readonline",
    "Sapiens": "https://drive.google.com/file/d/1x0dMOcDoaIIhGYglaLn5U5RvCO_-1tFQ/view",
    "Sherlock Holmes": "https://www.gutenberg.org/cache/epub/48320/pg48320-images.html",
    "The Martian": "https://en.readanybook.com/online/565265",
    "Ready Player One": "https://cdn.bookey.app/files/pdf/book/en/ready-player-one.pdf",
    "Ender's Game": "https://www.readanybook.com/online/565842",
    "Dune": "https://drive.google.com/file/d/1fb4vEpp_Up32hGRHWHaBdvDrsCz8aVzi/view",
    "Neuromancer": "https://www.readanybook.com/online/753746",
    "Snow Crash": "https://cdn.bookey.app/files/pdf/book/en/snow-crash.pdf",
    "The Girl with the Dragon Tattoo": "https://drive.google.com/file/d/0By9wjQndHrkTSFgtVnlWalhfRU0/edit?resourcekey=0-HZ6CCU_Df05mrNr8_NMQcQ",
    "Big Little Lies": "https://cdn.bookey.app/files/pdf/book/en/big-little-lies.pdf",
    "Rebecca": "https://archive.org/details/rebecca-daphne-du-maurier/page/n205/mode/2up",
    "Gone Girl": "https://drive.google.com/file/d/1esXkVoQLG7cGF_jVH5_yKSHUThOBrP61/view",
    "In The Woods": "https://i0.wp.com/tlbranson.com/wp-content/uploads/2021/11/In-the-Woods-Tana-French-Books-in-Order.jpg",
    "Shutter Island": "https://cdn.bookey.app/files/pdf/book/en/shutter-island.pdf",
    "The Silent Patient": "https://online.fliphtml5.com/ovtbw/amdg/#p=44",
    "The Hound of the Baskervilles": "https://www.scribd.com/document/845926940/The-Hound-of-the-Baskervilles",
    "Me Before You": "https://drive.google.com/file/d/0B7EbuVRnOZ7YWDZQR0lseTB4a1U/view?resourcekey=0-KtA1bnLS8ew2hnMzhs_4PQ",
    "Outlander": "https://www.readanybook.com/online/665097",
    "The Notebook": "https://docs.google.com/file/d/0B3vyNXp6qDWwOVRWTDJFdmNyYUk/view?resourcekey=0-5sQvu9p-L-v8Ss9Y16GPZw",
    "Jane Eyre": "https://www.gutenberg.org/cache/epub/1260/pg1260-images.html",
    "The Time Traveller's Wife": "https://cdn.bookey.app/files/pdf/book/en/the-time-traveler%27s-wife.pdf",
    "Twilight": "https://drive.google.com/file/d/0Bz2Q6tnPH2uiLUdDa1FMRXZpd1U/edit?resourcekey=0-yGZq9Xt6LHlpO-TDo1wX0Q",
    "The Fault in Our Stars": "https://pnu.edu.ua/depart/Inmov/resource/file/samostijna_robota/Green_John_The_Fault_in_Our_Stars.pdf",
    "The Da Vinci Code": "https://drive.google.com/file/d/0B1ySOrPxOWJmR212UnhlcWExVmc/view?resourcekey=0-RmsTa8ZuvpGK5qy9_sjIWg",
    "Altered Carbon": "https://ia601208.us.archive.org/14/items/calibre_library_71.196.137.201/Altered%20Carbon%20-%20Richard%20Morgan_3.pdf",
    "The Left Hand of Darkness": "https://www.mlook.mobi/files/month_1203/80e49eb29e78d387d11eb2927ebb8b0dff842941.pdf",
    "Atomic Habits": "https://drive.google.com/file/d/1eAZMdXO-Zn4_90TV365KMpIqfxUy7J0t/view",
    "The Power of Now": "https://dn790003.ca.archive.org/0/items/ThePowerOfNowEckhartTolle_201806/The%20Power%20Of%20Now%20-%20Eckhart%20Tolle.pdf",
    "Think and Grow Rich": "https://drive.google.com/file/d/0BwEKYdmMaJIGOEp6enBLM3VxRjg/view?resourcekey=0-oleVde4shCeqr-psMzy7oA",
    "The 5 AM Club": "https://files.addictbooks.com/wp-content/uploads/2022/11/The-5-AM-Club.pdf",
    "Deep Work": "https://cdn.bookey.app/files/pdf/book/en/deep-work.pdf"
}
df['pdf_link'] = df['Book'].map(pdf_links)

# Sidebar: Input sliders
st.sidebar.image("bookLogo.png", width=120)
st.sidebar.title("📖 Choose Your Preferences")
st.sidebar.markdown("<div class='book-meta'>Language: English only</div>", unsafe_allow_html=True)

with st.sidebar.expander("🎚️ Adjust Your Preferences", expanded=True):
    fiction = st.slider("Fiction", 1, 10, 5)
    mystery = st.slider("Mystery", 1, 10, 5)
    scifi = st.slider("Science Fiction", 1, 10, 5)
    romance = st.slider("Romance", 1, 10, 5)
    nonfiction = st.slider("Non-Fiction", 1, 10, 5)
    age = st.slider("Your Age", 15, 60, 25)
    mood = st.slider("Mood (1=sad, 10=happy)", 1, 10, 5)

# Recommendation logic
def get_recommendations(df, user_input):
    features = ['Fiction', 'Mystery', 'SciFi', 'Romance', 'Age', 'Mood', 'NonFiction']
    distances = euclidean_distances(df[features], user_input).flatten()
    df['Distance'] = distances
    return df.sort_values(by='Distance').reset_index(drop=True)

user_input = [[fiction, mystery, scifi, romance, age, mood, nonfiction]]
features = ['Fiction', 'Mystery', 'SciFi', 'Romance', 'Age', 'Mood', 'NonFiction']
X = df[features]

# Create a label just for classification purposes
df['Category'] = df['Fiction'].apply(lambda x: 'Fiction Lover' if x > 6 else 'Balanced Reader')
y = df['Category']

# Train DTC
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict user category
user_type = clf.predict(user_input)[0]
st.sidebar.markdown(f"<div class='book-meta'><b>📘 You are classified as:</b> <span style='color:#ef4444'>{user_type}</span></div>", unsafe_allow_html=True)
top_books = get_recommendations(df, user_input)

# Show top recommendation
best_book = top_books.iloc[0]
st.subheader("We found your perfect book match – enjoy! 📖")
st.markdown(f"""
<div class="book-box">
    <img class="header-img" src="{best_book['image_url']}" alt="{best_book['Book']} cover"/>
    <h2 class="book-title">{best_book['Book']}</h2>
    <div class="book-description">
        {best_book['Description']}
    </div>
    <div class="book-meta"><b>Language:</b> {best_book['Language']}</div>
    <div class="book-meta"><b>Genre Scores:</b></div>
    <div>
        <span class="genre-badge">Fiction: {best_book['Fiction']}</span>
        <span class="genre-badge">My
        stery: {best_book['Mystery']}</span>
        <span class="genre-badge">Sci-Fi: {best_book['SciFi']}</span>
        <span class="genre-badge">Romance: {best_book['Romance']}</span>
        <span class="genre-badge">Non-Fiction: {best_book['NonFiction']}</span>
    </div>
    <a href="{best_book['pdf_link']}" class="pdf-link" target="_blank">📖 Read PDF</a>
</div>
""", unsafe_allow_html=True)

# More recommendations
if st.button("🔄 Show More Recommendations"):
    st.subheader("📚 More Books You Might Like:")
    for _, row in top_books.iloc[1:4].iterrows():
        st.markdown(f"""
        <div class="book-box">
            <img class="header-img" src="{row['image_url']}" alt="{row['Book']} cover"/>
            <h3 class="book-title">{row['Book']}</h3>
            <div class="book-description">
                {row['Description']}
            </div>
            <div class="book-meta"><b>Language:</b> {row['Language']}</div>
            <div class="book-meta"><b>Genre Scores:</b></div>
            <div>
                <span class="genre-badge">Fiction: {row['Fiction']}</span>
                <span class="genre-badge">Mystery: {row['Mystery']}</span>
                <span class="genre-badge">Sci-Fi: {row['SciFi']}</span>
                <span class="genre-badge">Romance: {row['Romance']}</span>
                <span class="genre-badge">Non-Fiction: {row['NonFiction']}</span>
            </div>
            <a href="{row['pdf_link']}" class="pdf-link" target="_blank">📖 Read PDF</a>
        </div>
        """, unsafe_allow_html=True)
