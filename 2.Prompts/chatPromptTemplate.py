from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_prompt_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant that summarizes text."),
    ('human', "Please summarize the following text: {text}")
])

prompt = chat_prompt_template.invoke({
    "domain": "research",
    "text": "LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs and enables the creation of chains, which are sequences of calls to LLMs or other utilities."
})

print(prompt)