# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template="You are a professional report writer. You will be given a text and you will write a concise report summary about it in 5 lines. Here is the text: {text}"
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Attack on Titan"})
print(result)