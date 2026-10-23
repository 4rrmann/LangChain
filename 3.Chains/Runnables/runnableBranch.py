'''RunnableBranch is a control flow component in LangChain that allows you to conditionally route input data to different chains or runnables based on custom logic.

It functions like if/elif/else block for chains, where you define a set of condition functions, each associated with a runnable (eg: LLM call, prompt chain, or tool). The first matching condition is executed. If no condition matches, a default runnable is used (if provided)'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template='write the Character development about {chr} from AOT, quite brief in a single paragraph',
    input_variables=['chr']
)

model = ChatGroq(
    model='openai/gpt-oss-120b'
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Summarize the following context \n {context}',
    input_variables=['context']
)

chr_dev_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 150, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


# final_chain = RunnableSequence(chr_dev_chain, branch_chain)
final_chain = chr_dev_chain | branch_chain # LCEL

result = final_chain.invoke({'chr': 'Levi'})

print(result)