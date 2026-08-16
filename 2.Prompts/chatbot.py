from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

groq = ChatGroq(
    model="llama-3.3-70b-versatile"
    )

chat_history = [
    SystemMessage(content="You are a helpful assistant that summarizes text.")
]

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == 'exit':
        break
        
    result = groq.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Assistant: ",result.content)