from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

load_dotenv()

groq = ChatGroq(
    model="llama-3.3-70b-versatile"
)

st.header("Research Assistant")

paper_input_prompt = st.selectbox("Select Research paper Name", ["Select..", "Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "Neural Ordinary Differential Equations", "A Survey on Deep Learning in Medical Image Analysis", "Deep Residual Learning for Image Recognition", "Generative Adversarial Networks", "U-Net: Convolutional Networks for Biomedical Image Segmentation"])

style_input_prompt = st.selectbox("Select Style", ["Summarize", "Explain", "Critique", "Compare", "Contrast", "Evaluate", "Interpret", "Analyze"])

length_input_prompt = st.selectbox("Select Length", ["Short", "Medium", "Long"])

#template
template = PromptTemplate(
    template="Please {style_input_prompt} the research paper '{paper_input_prompt}' in a {length_input_prompt} length.",
    
    input_variables=["paper_input_prompt", "style_input_prompt", "length_input_prompt"],
)

#fill the placeholders in the template with the user inputs
prompt = template.invoke({
    "paper_input_prompt": paper_input_prompt,
    "style_input_prompt": style_input_prompt,
    "length_input_prompt": length_input_prompt
})

if st.button("Summarize"):
    st.text("Processing your request...")   

    result = groq.invoke(prompt)

    st.write(result.content)