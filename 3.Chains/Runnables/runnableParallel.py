'''RunnableParallel is a runnable primitive that allows multiple runnables to execute in parallel.

Each runnable receives the same input and processes it independently, producing a dictionary of outputs'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Expalin AOT's {emotions} emotions in quite brief paragraph",
    input_variables=['emotions']
)

prompt2 = PromptTemplate(
    template="Expalin Death Note's {emotions} emotions in quite brief paragraph",
    input_variables=['emotions']
)

model = ChatGroq(
    model='openai/gpt-oss-120b'
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "AOT": RunnableSequence(prompt1, model, parser),
    "DN": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"emotions": "Dark"})

print(result['AOT'], "\n")
print(result['DN'])