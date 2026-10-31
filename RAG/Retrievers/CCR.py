from langchain_core.documents import Document

doc1 = Document(page_content=(
    "Eren, Mikasa and Armin graduated from the 104th Training Corps after three years of harsh drills under Keith Shadis. "
    "Sasha was famous for stealing a potato during training, and Jean ranked sixth in the class. "
    "On the day of the graduation ceremony, the Colossal Titan returned and blasted a hole through the gate of Trost District. "
    "Following Armin's plan, Eren transformed into a Titan, lifted a giant boulder and sealed the hole in the wall. "
    "Afterward, Marco's death left Jean heartbroken, and the surviving cadets had to choose which branch of the military to join."
))

doc2 = Document(page_content=(
    "Annie Leonhart is the Female Titan, a cold and solitary soldier trained by her father in a special fighting style. "
    "During the 57th expedition she killed the members of Levi's squad in the forest. "
    "In Stohess she fought Eren through the city and then sealed herself inside a crystal. "
    "Years later she woke up and joined the Alliance to stop the Rumbling."
))

doc3 = Document(page_content=(
    "Levi Ackerman grew up in the Underground, where he was raised by Kenny Ackerman. "
    "He is obsessed with cleanliness and is known as humanity's strongest soldier. "
    "He lost his friends Isabel and Farlan, and later lost his entire squad and Erwin. "
    "In the final war he was badly injured but survived, carrying the memory of everyone he lost."
))

doc4 = Document(page_content=(
    "Armin Arlert is a gentle boy who dreams of seeing the ocean and reading forbidden books about the outside world. "
    "He is a brilliant strategist who devised the plan at Trost: Eren, in Titan form, would carry a boulder to plug the broken gate. "
    "Later he inherited the Colossal Titan from Bertholdt and became a peace ambassador after the war."
))

documents = [doc1, doc2, doc3, doc4]



from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

query = "How did Eren seal the hole in the wall at Trost?"
results = compression_retriever.invoke(query)

for d in results:
    print("-", d.page_content)