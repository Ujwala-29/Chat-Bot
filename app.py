import streamlit as st
from ctransformers import AutoModelForCausalLM
import os

def getLLamaresponse(input_text, no_words, blog_style):
    model_path = './models/llama-2-7b-chat.ggmlv3.q8_0.bin'  # Ensure this path is correct

    # Check if the model path exists
    if not os.path.exists(model_path):
        return f"Error: The model file does not exist at the specified path: {model_path}"

    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Define the prompt
    prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."

    # Generate the response from the model
    response = model.generate(input_text=prompt, max_length=256)

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
