import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

def getLLamaresponse(input_text, no_words, blog_style):
    # Use Hugging Face model directly
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Hugging Face model ID
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the prompt
    prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."

    # Tokenize the input and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=256)

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Streamlit UI setup
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional 2 fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    if input_text and no_words and blog_style:
        st.write(getLLamaresponse(input_text, no_words, blog_style))
    else:
        st.warning("Please fill in all the fields.")
