from langchain_core.documents import Document

doc1 = Document(page_content="Eren Yeager is the protagonist of Attack on Titan. After his mother is eaten by a Titan, he vows to destroy every Titan on Earth.")
doc2 = Document(page_content="Eren Yeager is the main hero of Attack on Titan. He swears revenge on all Titans after watching his mother die in Shiganshina.")
doc3 = Document(page_content="Eren Yeager, the main character of Attack on Titan, inherits the Attack Titan and later the Founding Titan, which he uses to start the Rumbling.")
doc4 = Document(page_content="Levi Ackerman is humanity's strongest soldier and a captain in the Survey Corps. He is famous for his speed and his obsession with cleanliness.")
doc5 = Document(page_content="Captain Levi Ackerman of the Survey Corps is considered the strongest fighter alive. He is sharp-tongued, tidy and extremely skilled with his blades.")
doc6 = Document(page_content="Armin Arlert is a gentle but brilliant strategist who dreams of seeing the ocean. His plans save the Survey Corps many times, and he inherits the Colossal Titan.")
doc7 = Document(page_content="Historia Reiss is the hidden heir of the royal family. She spends years as Krista Lenz before taking the throne and choosing to rule for her people.")
doc8 = Document(page_content="Reiner Braun is the Armored Titan and a Marleyan Warrior who infiltrates the walls. He is haunted by guilt over destroying Wall Maria.")

documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]



from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.5},
)

query = "Tell me about Eren Yeager"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content)