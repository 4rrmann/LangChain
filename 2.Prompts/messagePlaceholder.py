#A MessagePlacehoder in LangChain is a special placeholder used inside a ChatPromptTemplate to dynamically insert chat history of a list of messages at runtime.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#ChatTemplate with a MessagesPlaceholder
chat_prompt_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}'),
])

chat_history = []
#load chat history with messages
with open("chat_history.txt", "r") as f:
        chat_history.extend(f.readlines())

print(chat_history)


#Create a prompt with the chat history and a user's new query
prompt = chat_prompt_template.invoke({'chat_history': chat_history, 'query': "Where's my refund?"})

print(prompt)