from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template="Who is Armin Arlert, quite brief in a single paragraph? - \n {context}",
    input_variables=['context']
)

parser = StrOutputParser()


loader = CSVLoader(file_path='AOTcharacters.csv')

docs = loader.load()


print("Total Pages Contents: ",len(docs))

print("\nPage Content:\n",docs[2].page_content)
print("\nMetadata:\n",docs[2].metadata)



chain = prompt | model | parser

result = chain.invoke({'context':docs[0].page_content})
print("\n",result)