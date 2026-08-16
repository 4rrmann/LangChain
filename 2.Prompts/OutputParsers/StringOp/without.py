# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
# )

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)
# model = ChatHuggingFace(llm=llm)

#1st Prompt (detailed report)
template1 = PromptTemplate(
    input_variables=["topic"],
    template="You are a professional report writer. You will be given a topic and you will write a detailed report about it. Here is the topic: {topic}"
)

#2nd Prompt (concise report summary)
template2 = PromptTemplate(
    input_variables=["text"],
    template="You are a professional report writer. You will be given a text and you will write a concise report summary about it. Here is the text: {text}"
)

prompt1 = template1.invoke({'topic': "Attack on Titan"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

# print(result1.content + "\n")
print(result2.content)