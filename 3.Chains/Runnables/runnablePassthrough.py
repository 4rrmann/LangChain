'''RunnablePassthrough is a special primitive that simply returns the input as output without modifying it'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

passthrough = RunnablePassthrough()
print(passthrough.invoke("RunnablePassthrough is a special primitive that simply, Returns the input as output without modifying it\n"))

prompt1 = PromptTemplate(
    template='write the Character development about {chr} from AOT, quite brief in a single paragraph',
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

chr_dev_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'chr': RunnablePassthrough(),
    '1 line': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(chr_dev_chain, parallel_chain)

print(final_chain.invoke({'chr': 'Levi'}))