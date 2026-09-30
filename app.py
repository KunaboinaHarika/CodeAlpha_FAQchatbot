%%writefile app.py
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI FAQ Assistance Chatbot")
st.caption("Ask questions about the CodeAlpha Artificial Intelligence Internship!")

# 1. Dataset of FAQ Questions & Answers
FAQ_DATA = [
    {"question": "What is CodeAlpha?", "answer": "CodeAlpha is a software development company providing hands-on internship programs in emerging technologies."},
    {"question": "How long is the internship program?", "answer": "The internship program runs for 1 month with flexible working hours."},
    {"question": "Will I get a certificate upon completion?", "answer": "Yes, you will receive a QR-verified Completion Certificate upon submitting your completed tasks."},
    {"question": "How many tasks should I complete?", "answer": "You need to complete any 2 or 3 tasks listed in your domain task list."},
    {"question": "What is the GitHub repository naming rule?", "answer": "Repositories must follow the exact structure: CodeAlpha_ProjectName."},
    {"question": "How do I submit my completed project?", "answer": "Upload code to GitHub, share a demo video on LinkedIn tagging @CodeAlpha, and fill out the Submission Form."}
]

df = pd.DataFrame(FAQ_DATA)

# 2. Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered)

df['processed'] = df['question'].apply(preprocess_text)

# 3. Vectorization & Similarity Engine
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

def get_faq_response(user_query):
    processed_query = preprocess_text(user_query)
    if not processed_query.strip():
        return "I couldn't understand that. Please try asking again!"
    
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    best_match_idx = similarities.argmax()
    confidence = similarities[best_match_idx]
    
    if confidence < 0.2:
        return "I don't have enough details on that specific query. Please consult the CodeAlpha guidelines."
    
    return df.iloc[best_match_idx]['answer']

# 4. Streamlit Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me any question about your CodeAlpha internship."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = get_faq_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
