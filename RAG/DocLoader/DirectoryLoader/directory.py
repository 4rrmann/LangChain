from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template="What kind of emotion it holds, quite brief in a single paragraph? - \n {context}",
    input_variables=['context']
)

parser = StrOutputParser()



loader = DirectoryLoader(
    path='AOTseasonsSummary',
    glob='*.pdf', #fileType
    loader_cls=PyPDFLoader
)

# Loads everything at once.
docs = loader.load()

# Loads on demand.
# docs = loader.lazy_load()

print("Total Pages in All Docs: ",len(docs))

print("\nPage Content:\n",docs[0].page_content)
print("\nMetadata:\n",docs[0].metadata)



chain = prompt | model | parser

result = chain.invoke({'context':docs[0].page_content})
print("\n",result)