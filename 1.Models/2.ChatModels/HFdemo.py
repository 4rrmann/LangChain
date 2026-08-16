from langchian_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational",
    provider="groq"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is GenAI?")

print(result.content)