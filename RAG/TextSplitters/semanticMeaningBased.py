from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample_text = '''
Humanity lives inside three giant walls called Wall Maria, Wall Rose and Wall Sina. These walls protect people from the Titans, huge humanoid creatures that devour humans. Titans have no need to eat, yet they hunt people relentlessly. Their only weakness is the nape of the neck, and soldiers cut it using 3D Maneuver Gear. The Survey Corps explores the land beyond the walls, while the Garrison guards the gates and the Military Police protects the king.
When Eren Yeager was ten, the Colossal Titan kicked open the gate of Shiganshina District. His mother Carla was trapped under their collapsed house and was eaten before his eyes. Eren swore to destroy every Titan on Earth. His adoptive sister Mikasa and his friend Armin escaped with him to Wall Rose. Their hometown was lost, and refugees faced hunger and despair.
Five years later, Eren, Mikasa and Armin joined the 104th Training Corps. Instructor Keith Shadis pushed the cadets through brutal drills with swords and grappling gear. Reiner, Bertholdt, Annie, Jean, Connie, Sasha and Krista became their closest comrades. On the day of graduation, the Colossal Titan appeared again at Trost and destroyed the gate. In the chaos, Eren was swallowed by a Titan but emerged as a Titan himself. Using Armin's plan, Eren carried a boulder and sealed the hole in the wall.

Soon the Survey Corps discovered an intelligent Titan hunting Eren outside the walls. This Female Titan killed Captain Levi's elite squad in the forest. Armin deduced that the Female Titan was a cadet, and suspicion fell on Annie Leonhart. In Stohess, Annie transformed and fought Eren through the city streets. When she was cornered, she sealed herself inside a crystal and refused to speak.
Years later, the story moved across the sea to the nation of Marley, where Eldians were treated as devils. Eren, now the Founding Titan, unleashed the Rumbling, sending countless Colossal Titans to flatten the world. Mikasa, Armin, Levi and the former Warriors formed an alliance to stop him. In the end, Mikasa ended Eren's life, and the power of the Titans vanished forever.
'''

docs = text_splitter.create_documents([sample_text])
print(len(docs))
print(docs)