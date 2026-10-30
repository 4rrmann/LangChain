from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Define the query
query = "Attack on Titan main characters"

# Retrieve relevant Wikipedia documents
docs = retriever.invoke(query)

# Display the retrieved documents
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)