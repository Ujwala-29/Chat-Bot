import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Set up Streamlit
st.set_page_config(page_title="Zephyr Chatbot", layout="centered")
st.title("🧠 Zephyr-7B Chatbot")

# Load model and tokenizer only once
@st.cache_resource
def load_zephyr_model():
    model_id = "HuggingFaceH4/zephyr-7b-alpha"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the pipeline
generator = load_zephyr_model()

# Chat input
user_input = st.text_input("👤 You:", placeholder="Ask me anything...")

# Chat logic
if user_input:
    with st.spinner("Thinking..."):
        prompt = f"<|system|>You are a helpful assistant.<|user|>{user_input}<|assistant|>"
        result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        answer = result[0]["generated_text"].split("<|assistant|>")[-1].strip()
        st.markdown(f"**🤖 Zephyr:** {answer}")
