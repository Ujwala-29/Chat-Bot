import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLaMA 2 model

def getLLamaresponse(input_text, no_words, blog_style):

    ### LLaMA 2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt Template
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    ## Generate the response from the LLaMA 2 model
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = llm(formatted_prompt)
    print(response)
    return response

# Streamlit UI Setup
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# User input fields
input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
    if no_words.isdigit():  # Ensures the input is numeric
        no_words = int(no_words)
    else:
        no_words = 0

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    if no_words > 0:
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.warning("Please enter a valid number for the word count.")
