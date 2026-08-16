from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

from dotenv import load_dotenv
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

schema = [
    ResponseSchema(name='fact1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()},
    template='Give 3 fact about {topic} \n {format_instruction}'
)

# prompt = template.invoke({'topic': 'Death Note'})
# result = model.invoke(prompt)

# fresult = parser.parse(result.content)
# print(fresult)

chain = template | model | parser
result = chain.invoke({'topic': 'Death Note'})
print(result)

'''
Limitations of StructureOutputParser: No data validation

A basic `StructuredOutputParser` in frameworks like LangChain translates prompt instructions into an expected format, but it checks only basic syntax and keys. It lacks deep semantic and constraint validation, meaning models can still output logically flawed, out-of-range, or semantically invalid values that pass the structural check.
'''