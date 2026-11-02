from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()



# Indexing (1. Document Ingestion)
video_id = "vDIFuEamH5c"
try:
    ytt_api = YouTubeTranscriptApi()

    # with Timestamps
    transcript_list = ytt_api.fetch(
    video_id,
    languages=["en"]
)

    # no Timestamps, flatten it to plain text
    transcript_text = " ".join(
    snippet.text for snippet in transcript_list
)

    print("Transcript Text:\n",transcript_text)

except TranscriptsDisabled:
    print("No captions available for this video")



# Indexing (2. Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_text])

# print((len(chunks)))



# Indexing (3. Embedding Generation & 4. Store in Vector Store)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['a03595e4-4662-4f41-ba68-1b18284720cd']))



# Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
print(retriever)

query = "Who is Eren?"
result = retriever.invoke(query)
# print(result)



# Augumentation
llm = ChatGroq(
    model='openai/gpt-oss-120b'
)

prompt = PromptTemplate(
    template='''
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say: you don't know
        
        {context}
        Question: {question}
        ''',
        input_variables= ['context', 'question']
)

question = "is the topic of titans discussed in this video? if yes then what was discussed?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)



# Generation:
answer = llm.invoke(final_prompt)
# print("\nFinal Answer:",answer.content)





# Chains:
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

paralle_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

paralle_chain.invoke("Who is Eren?") # Context & Question

parser = StrOutputParser()
main_chain = paralle_chain | prompt | llm | parser

query = "\n\n[ Can you summarize the video? ]"
main_result = main_chain.invoke(query)

print(query)
print("-> ",main_result)