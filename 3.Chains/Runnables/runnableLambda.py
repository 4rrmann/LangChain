'''RunnableLambda is a runnable primitive that allows you to apply custom Python functions within an AI pipeline.

It acts as a middleware between different AI components, enabling preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)
print("WordCount: ",runnable_word_counter.invoke("Count my words"))

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
    # 'word_count': RunnableLambda(lambda x: len(x.split()))
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(chr_dev_chain, parallel_chain)

result = final_chain.invoke({'chr': 'Levi'})

final_result = """{} \n Word Count: {}""".format(result['chr'], result['word_count'])
print(final_result)