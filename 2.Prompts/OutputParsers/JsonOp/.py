# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
# )

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)
# model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()

template = PromptTemplate(
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()},
    # template="Give me the name, age and city of a fictional person \n {format_instruction}"
    template='Give me 5 facts about {topic} \n {format_instruction}'
)

# prompt = template.format()
# print(f"Prompt: {prompt}\n")

# result = model.invoke(prompt)
# # print(result)

# fresult = parser.parse(result.content)
# print(f"Final Result: {fresult}\n")
# print(fresult['name'])
# print(type(fresult))


chain = template | model | parser
result = chain.invoke({'topic': 'AOT'})

print(result)

'''
The core `Limitation` of LangChain's `JsonOutputParser` is that it only ensures the model's output is syntactically valid JSON, but it does not enforce a strict internal schema or validate data types. It returns a raw Python dictionary without checking if required keys exist or if values match expected formats, meaning keys can be missing, renamed, or mistyped by the LLM.
'''