import streamlit as st
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Log in using your Hugging Face token
# Login to Hugging Face once, ensure to use a secure token
login('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')  # Replace with your Hugging Face token

# Load pre-trained model and tokenizer globally
@st.cache_resource
def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Model path in HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # If CUDA is available, move the model to GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model loaded on GPU.")
    else:
        print("Model loaded on CPU.")
    
    return model, tokenizer

# Initialize the model and tokenizer
model, tokenizer = load_model()

# Function to get LLM response
def get_llama_response(input_text, no_words, blog_style):
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # If using GPU, move input to GPU
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    # Generate output
    output = model.generate(input_ids, max_length=no_words, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return decoded_output

# Streamlit UI
def main():
    st.title("AI Blog Generator")

    # Input fields
    topic = st.text_input("Enter the Blog Topic", "llm")  # Default topic is "llm"
    no_words = st.slider("Number of Words", 50, 500, 100)  # Word count slider
    blog_style = st.selectbox("Select Blog Style", ["Common People", "Researchers"])

    # Generate button
    if st.button("Generate Blog"):
        if topic and no_words:
            try:
                result = get_llama_response(topic, no_words, blog_style)
                st.write(result)  # Display generated blog
            except Exception as e:
                st.error(f"Error generating blog: {str(e)}")

if __name__ == "__main__":
    main()
