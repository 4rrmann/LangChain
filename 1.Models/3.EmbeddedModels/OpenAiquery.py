from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OpenAIEmbeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimension=32
)

result = OpenAIEmbeddings.embed_query("Convert this in embedding vector")

print(result)