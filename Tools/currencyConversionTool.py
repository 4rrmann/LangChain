from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
print("API key loaded:", bool(os.getenv("GROQ_API_KEY")))

import json


#Tools Creation:
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency coversion factor between a given base currency and a target currency"""

    url = f"https://v6.exchangerate-api.com/v6/apikey?hehe/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Given a currency rate of this function calculates the target currency value from a given base currency value"""

    return base_currency_value*conversion_rate



conversion = get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'})
# print(conversion) # 94.8321

converter = convert.invoke({'base_currency_value': 10, 'conversion_rate':94.8321})
print(converter)



llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])


messages = [HumanMessage('What is the conversion factor between USD and INR, and based on that can you convert 10 USD to INR')]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
print("AI Message (tool_calls): ",ai_msg.tool_calls,"\n")


for tool_call in ai_msg.tool_calls:

    # Execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
        tool_msg1 = get_conversion_factor.invoke(tool_call)
        # fetch this conversion rate
        data = json.loads(tool_msg1.content)
        conversion_rate = data['conversion_rate']
        # append this tool message to messages list
        messages.append(tool_msg1)

    # Execute the 2nd tool using the conversion rate from tool 1
    elif tool_call['name'] == 'convert':
        if conversion_rate is None:
                raise ValueError(
                    "Fetch the conversion rate before calling convert."
                )
        #fetch the current arg
        tool_call['args']['conversion_rate'] = conversion_rate
        
        tool_msg2 = convert.invoke(tool_call)
        messages.append(tool_msg2)

print(messages,"\n\n")

result = llm_with_tools.invoke(messages)
print("Final Result: ",result)