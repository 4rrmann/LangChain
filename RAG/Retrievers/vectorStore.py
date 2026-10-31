from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Source Docs
documents = [

    Document(
    page_content="Eren Yeager is the protagonist of Attack on Titan. After the Colossal Titan destroys his hometown and his mother is eaten, he vows to wipe out every Titan. He inherits the Attack Titan and later the Founding Titan, and his hunger for freedom drives him to start the Rumbling."),

    Document(
    page_content="Mikasa Ackerman is Eren's adoptive sister and one of the strongest soldiers of her generation. A member of the Ackerman clan, she is fiercely loyal to Eren and always carries the scarf he gave her. In the end, she is the one who stops him."),

    Document(
    page_content="Armin Arlert is a brilliant strategist whose plans save the Survey Corps again and again, from Trost to Shiganshina. Gentle and self-doubting, he dreams of seeing the ocean. He inherits the Colossal Titan from Bertholdt and later works to bring peace to the world."),

    Document(
    page_content="Levi Ackerman is known as humanity's strongest soldier and leads the Survey Corps' Special Operations Squad. Raised in the Underground, he is obsessed with cleanliness and hides a caring heart behind a sharp tongue. He survives the war, carrying the memory of the comrades he lost."),

    Document(
    page_content="Reiner Braun is the Armored Titan and a Marleyan Warrior who infiltrates the walls as a cadet. He lives as both soldier and enemy and is haunted by guilt for destroying Wall Maria. In the final season, he fights alongside his former enemies to stop the Rumbling.")

]


# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="aots_collection"
)


# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})


query = "Who's the Ackerman?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content)