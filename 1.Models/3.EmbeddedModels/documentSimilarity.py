from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Attack on Titan is a thrilled anime series that follows the story of Eren Yeager and his friends as they fight against giant humanoid creatures known as Titans.",
    "Demon Slayer is an action-packed anime series that follows the story of Tanjiro Kamado, a young boy who becomes a demon slayer to save his sister and avenge his family.",
    "Death Note is a psychological thriller anime series that follows the story of Light Yagami, a high school student who gains the power to kill anyone whose name he writes in a supernatural notebook.",
    "Naruto is a long-running anime series that follows the story of Naruto Uzumaki, a young ninja who dreams of becoming the strongest ninja in his village and earning the respect of his peers.",
    "One Piece is a pirate adventure anime series that follows the story of Monkey D. Luffy and his crew as they search for the legendary treasure known as One Piece in order to become the Pirate King.",
]

query = "What is the story of Death Note?"

docs_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity between the query and each document
scores = cosine_similarity([query_embedding], docs_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]  # Print the index and score of the most similar document

print(f"\n{query} \n{documents[index]}  \nScore: {score}")