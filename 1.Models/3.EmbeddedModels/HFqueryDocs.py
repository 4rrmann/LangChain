from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "This is a sample text to be embedded."
# docs = [
#     "This is the first document to be embedded.",
#     "This is the second document to be embedded.",
#     "This is the third document to be embedded."
# ]

vector = embeddings.embed_query(text)
# vector = embeddings.embed_documents(docs)

print(str(vector))