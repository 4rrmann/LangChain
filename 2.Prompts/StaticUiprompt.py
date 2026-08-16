from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq = ChatGroq(
    model="llama-3.3-70b-versatile"
)

st.header("Research Assistant")

user_input_prompt = st.text_input("Enter your prompt:")

if st.button("Summarize"):
    st.text("Processing your request...")

    result = groq.invoke(user_input_prompt)

    st.write(result.content)