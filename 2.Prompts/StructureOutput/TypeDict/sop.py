from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

#Schema
class ReviewSchema(TypedDict):

    summary: Annotated[str, "A concise summary of the text provided."]
    sentiment: Annotated[Literal["pos", "neg", "neutral"], "The sentiment of the text provided."]
    key_themes: Annotated[list[str], "The main themes or topics discussed in the text provided."]

    pros: Annotated[Optional[str], "The positive aspects of the text provided."]
    cons: Annotated[Optional[str], "The negative aspects of the text provided."]

structured_model = model.with_structured_output(ReviewSchema)

result = structured_model.invoke("Attack on Titan is a dark fantasy anime masterpiece. It follows humanity trapped inside giant walls by man-eating humanoids called Titans. The story begins as a survival horror show. It grows into a complex political thriller about freedom, war, and human nature. The plot twists shock viewers completely. The animation by Wit Studio and MAPPA is stunning. The action scenes are fast and intense. The music by Hiroyuki Sawano and Kohta Yamamoto is epic. Some fans dislike the final ending arc, but it remains a cultural milestone with deep themes." \
"The protagonist Eren Yeager swears to wipe out every Titan after his mother is killed during the breach of Wall Maria. Alongside Mikasa and Armin, Eren joins the Scout Regiment. They uncover secrets about the origins of the Titans and their enemies across the ocean. This turns a simple battle for survival into a morally ambiguous global conflict.")

print(result)
print(result['summary'])
print(result['sentiment'])