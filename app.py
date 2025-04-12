import streamlit as st
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Log in using your Hugging Face token
login('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')  # Replace with your Hugging Face token

# Function to get LLM response
def getLLamaresponse(input_text, no_words, blog_style):
    # Load pre-trained model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this model name to your own Hugging Face model path
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate output
    output = model.generate(input_ids, max_length=no_words, num_return_sequences=1, temperature=0.7)
    
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
                result = getLLamaresponse(topic, no_words, blog_style)
                st.write(result)  # Display generated blog
            except Exception as e:
                st.error(f"Error generating blog: {str(e)}")

if __name__ == "__main__":
    main()
