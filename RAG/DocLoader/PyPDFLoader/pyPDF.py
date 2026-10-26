from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template="What's the Aim of this character, quite brief in a single paragraph? - \n {chr}",
    input_variables=['chr']
)

parser = StrOutputParser()



loader = PyPDFLoader('AOTCharacters.pdf')
docs = loader.load()

print(len(docs), "\n") #Pages

print("Page Content:\n",docs[2].page_content)
print("Metadata:\n",docs[1].metadata, "\n")



chain = prompt | model | parser
result = chain.invoke({'chr':docs[1].page_content})

print(result)