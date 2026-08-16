from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq = ChatGroq(
    model="llama-3.3-70b-versatile"
)

messages = [
    SystemMessage(content="You are a helpful assistant that summarizes text."),
    HumanMessage(content="Please summarize the following text: 'LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs and enables the creation of chains, which are sequences of calls to LLMs or other utilities.'")
]

result = groq.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)