import streamlit as st
import pandas as pd
import spacy
import joblib
import os

# Load model and data
model = joblib.load("Models/tekbot_model.pkl")
vectorizer = joblib.load("Models/tekbot_vectorizer.pkl")
df = pd.read_csv("Data/intent_dataset_cleaned.csv")
nlp = spacy.load("en_core_web_sm")

# Build intent-response map
intent_responses = df.groupby("intent")["response"].first().to_dict()
intent_history = []

# Preprocess user input
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Generate follow-up questions
def generate_questions(response, n=2):
    doc = nlp(response)
    noun_phrases = list(set(chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1))
    templates = [
        "Can you tell me more about {}?",
        "How does {} work?",
        "Why is {} important?"
    ]
    follow_ups = []
    for i, np in enumerate(noun_phrases[:n]):
        template = templates[i % len(templates)]
        follow_ups.append(template.format(np))
    return follow_ups

# Bot logic
def get_bot_response(user_input):
    cleaned = preprocess(user_input)
    vect_input = vectorizer.transform([cleaned])
    predicted_intent = model.predict(vect_input)[0]
    intent_history.append(predicted_intent)
    response = intent_responses.get(predicted_intent, "Sorry, I don't have an answer for that.")
    follow_ups = generate_questions(response)
    return response, follow_ups

# Streamlit UI
st.set_page_config(page_title="TekBot", page_icon="🤖")
st.title("🤖 TekBot – Your Company Assistant")

user_input = st.text_input("Ask TekBot something about the company...")

if user_input:
    response, follow_ups = get_bot_response(user_input)
    st.markdown(f"**TekBot:** {response}")
    
    if follow_ups:
        st.markdown("**🔁 Follow-up suggestions:**")
        for q in follow_ups:
            st.write(f"👉 {q}")
    
    st.markdown("---")
    st.markdown(f"**🧠 Intent History:** {', '.join(intent_history)}")
