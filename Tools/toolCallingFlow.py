from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Tool:
@tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({"a":1, "b":6})
# print(result)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)


# Tool Binding:
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

llm_with_tools = llm.bind_tools([multiply])

print(llm_with_tools.invoke("hey"))
print("\n\n")

result = llm_with_tools.invoke(
    "You must call the multiply tool with a=6 and b=1."
    "Do not calculate the answer yourself."
)
print(result)
#Tool Calling
print("\n\n")
print(result.tool_calls[0])

#Tool Execution
print("\n\n")
print(multiply.invoke(result.tool_calls[0]))

query = HumanMessage("You must call the multiply tool with a=6 and b=1."
    "Do not calculate the answer yourself.")
messages = [query]

result = llm_with_tools.invoke(messages)
messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)
print(messages) # History (human+ai+tool msgs)

final_result = llm_with_tools.invoke(messages).content
print("\n\n",final_result)