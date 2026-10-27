from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template="Who is Erwin Smith, quite brief in a single paragraph? - \n {context}",
    input_variables=['context']
)

parser = StrOutputParser()


url = 'https://docs.google.com/document/d/1GJ_TkRCRNGyvtyyxxwF718MOh6oqzHH0G_K_R2s6yiM/edit?tab=t.0'
loader = WebBaseLoader(url)

docs = loader.load()


print("Total Pages Contents: ",len(docs))

print("\nPage Content:\n",docs[0].page_content)
print("\nMetadata:\n",docs[0].metadata)



chain = prompt | model | parser

result = chain.invoke({'context':docs[0].page_content})
print("\n",result)