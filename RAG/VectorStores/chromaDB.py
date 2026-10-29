from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


from langchain_core.documents import Document

# Create LangChain documents for AoT characters
doc1 = Document(
    page_content="Eren Yeager is the protagonist of Attack on Titan. After the Colossal Titan destroys his hometown and his mother is eaten, he vows to wipe out every Titan. He inherits the Attack Titan and later the Founding Titan, and his hunger for freedom drives him to start the Rumbling.",
    metadata={"faction": "Survey Corps"}
)

doc2 = Document(
    page_content="Mikasa Ackerman is Eren's adoptive sister and one of the strongest soldiers of her generation. A member of the Ackerman clan, she is fiercely loyal to Eren and always carries the scarf he gave her. In the end, she is the one who stops him.",
    metadata={"faction": "Survey Corps"}
)

doc3 = Document(
    page_content="Armin Arlert is a brilliant strategist whose plans save the Survey Corps again and again, from Trost to Shiganshina. Gentle and self-doubting, he dreams of seeing the ocean. He inherits the Colossal Titan from Bertholdt and later works to bring peace to the world.",
    metadata={"faction": "Survey Corps"}
)

doc4 = Document(
    page_content="Levi Ackerman is known as humanity's strongest soldier and leads the Survey Corps' Special Operations Squad. Raised in the Underground, he is obsessed with cleanliness and hides a caring heart behind a sharp tongue. He survives the war, carrying the memory of the comrades he lost.",
    metadata={"faction": "Survey Corps"}
)

doc5 = Document(
    page_content="Reiner Braun is the Armored Titan and a Marleyan Warrior who infiltrates the walls as a cadet. He lives as both soldier and enemy and is haunted by guilt for destroying Wall Maria. In the final season, he fights alongside his former enemies to stop the Rumbling.",
    metadata={"faction": "Marleyan Warriors"}
)


documents = [doc1, doc2, doc3, doc4, doc5]



vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)



# Add Documents:
'''print(vector_store.add_documents(documents))'''



# View Documents:
'''print(vector_store.get(include=['embeddings','documents','metadatas']))'''



# Search Documents:
'''print(vector_store.similarity_search(
    query='Who is Titan?',
    k=3 # No. of similar docs
))'''



# Search with Similarity Score:
'''print(vector_store.similarity_search_with_score(
    query='Who is Titan?',
    k=3
))'''



# Meta-data filtering:
'''print(vector_store.similarity_search_with_score(
    query="",
    filter={'faction': 'Marleyan Warriors'}
))'''



# Update documents:
'''updated_doc1 = Document(
    page_content="Goodbye! Reiner",
    metadata={"faction": "Marleyan Warriors"}
)

vector_store.update_document(document_id='e4582992-ab36-429a-a1dd-bc0a68a3f393', document=updated_doc1)'''



# Delete Document:
'''vector_store.delete(ids=['78b75d69-0599-40a7-9cf8-f18cb988624e'])'''



# View Documents:
print(vector_store.get(include=['embeddings','documents','metadatas']))