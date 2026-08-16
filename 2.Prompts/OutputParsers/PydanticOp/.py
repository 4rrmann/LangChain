# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of any AOT {place} Character \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place': 'Anime'})
# print(f"Prompt: {prompt}\n")

# result = model.invoke(prompt)

# fresult = parser.parse(result.content)
# print(fresult)

chain = template | model | parser
fresult = chain.invoke({'place': 'Anime'})
print(fresult)