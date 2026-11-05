from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_core.messages import HumanMessage

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun

searchtool = DuckDuckGoSearchRun()
# results = searchtool.invoke("AOT")
# print(results)


@tool
def get_weather_data(city: str) -> str:
    """This function fetches the current weather data for a given city"""

    url = f"https://api.weatherstack.com/current?access_key=61fdb7825d74577ebb4acbb36ff155d1&query={city}"

    response = requests.get(url)
    return response.json()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# print(llm.invoke('hey'))



# Pulls the standard ReAct agent prompt
# prompt = hub.pull("hwchase17/react", dangerously_pull_public_prompt=True)
from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)

# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[searchtool, get_weather_data],
    prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[searchtool, get_weather_data],
    verbose=True,
    handle_parsing_errors=True
)

# Invoke
response = agent_executor.invoke({'input': "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)

print("\n\n", response['output'])