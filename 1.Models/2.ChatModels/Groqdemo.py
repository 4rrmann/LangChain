from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

result = chat_model.invoke("Explain Neural Networks")

print(result)
print(result.content)