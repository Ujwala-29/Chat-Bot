import streamlit as st
from transformers import pipeline

# Title
st.title("🧠 Free AI Chatbot (FLAN-T5)")

# Initialize the model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_model()

# User input
user_input = st.text_input("You:", "")

# Response
if user_input:
    prompt = f"Q: {user_input}\nA:"
    response = generator(prompt, max_length=100, temperature=0.7)[0]['generated_text']
    st.text_area("AI:", response, height=150)
