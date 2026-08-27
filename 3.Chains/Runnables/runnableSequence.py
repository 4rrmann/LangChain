from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='write the Character development about {chr} from AOT',
    input_variables=['chr']
)

model = ChatGroq(
    model='openai/gpt-oss-120b'
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='1 line for that Character {context}',
    input_variables=['context']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'chr':'Levi'}))