from langchain_core.documents import Document

doc1 = Document(page_content="Bertholdt Hoover is the Colossal Titan. In the year 845 he kicked open the gate of Shiganshina and broke Wall Maria, letting the Titans flood in.")
doc2 = Document(page_content="Reiner Braun is the Armored Titan. He smashed the inner gate of Shiganshina District, which led to the fall of Wall Maria.")
doc3 = Document(page_content="Eren Yeager used the Founding Titan to start the Rumbling, calling the giant Titans hidden inside the walls to march out and flatten the world.")
doc4 = Document(page_content="The walls were built about a hundred years before the story begins by King Karl Fritz, using the power of the Founding Titan and hardened Colossal Titans.")
doc5 = Document(page_content="Mikasa Ackerman is the strongest soldier of her generation and Eren's adoptive sister. She is famous for her speed with the 3D Maneuver Gear.")
doc6 = Document(page_content="Armin Arlert is a gifted strategist who dreams of seeing the ocean. His plans, such as sealing the hole at Trost, save many lives.")
doc7 = Document(page_content="Levi Ackerman is a captain of the Survey Corps who is obsessed with cleanliness and is known as humanity's strongest soldier.")
doc8 = Document(page_content="Historia Reiss is the hidden heir of the royal family. She lives as Krista Lenz for years before taking the throne.")

documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]



from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)


llm = ChatGroq(
    model='openai/gpt-oss-120b'
)


# Retrievers:
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# Query:
query = "Who was responsible for the fall of Wall Maria, and what role did each Titan shifter play in the attack?"

# Retrieve Results:
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)

print("\n","*"*17)

for i, doc in enumerate(multiquery_results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)