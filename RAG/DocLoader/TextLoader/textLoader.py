from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template="What kind of emotions holds in this speech, quite brief in a single paragraph? - \n {speech}",
    input_variables=['speech']
)

parser = StrOutputParser()



loader = TextLoader('CommanderErwinSmithSpeech.txt', encoding="utf-8")

docs = loader.load()
# print(docs) # metadata & page_content

print(type(docs)) #list
print(len(docs)) #1

print(docs[0].page_content)
print(docs[0].metadata, "\n")



chain = prompt | model | parser
result = chain.invoke({'speech':docs[0].page_content})

print(result)