from langchain_groq import ChatGroq
from typing import TypedDict, Literal, Optional, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

#Schema
json_schema = {
    "title": "ReviewAnalysisSchema",
    "type": "object",
    "description": "A schema for analyzing and summarizing reviews, including sentiment and key themes.",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A concise summary of the text provided."
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neutral"],
            "description": "The sentiment of the text provided."
        },
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "The main themes or topics discussed in the text provided."
        },
        "pros": {
            "type": ["string", "null"],
            "description": "The positive aspects of the text provided."
        },
        "cons": {
            "type": ["string", "null"],
            "description": "The negative aspects of the text provided."
        }
    },
    "required": ["summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("Attack on Titan is a dark fantasy anime masterpiece. It follows humanity trapped inside giant walls by man-eating humanoids called Titans. The story begins as a survival horror show. It grows into a complex political thriller about freedom, war, and human nature. The plot twists shock viewers completely. The animation by Wit Studio and MAPPA is stunning. The action scenes are fast and intense. The music by Hiroyuki Sawano and Kohta Yamamoto is epic. Some fans dislike the final ending arc, but it remains a cultural milestone with deep themes." \
"The protagonist Eren Yeager swears to wipe out every Titan after his mother is killed during the breach of Wall Maria. Alongside Mikasa and Armin, Eren joins the Scout Regiment. They uncover secrets about the origins of the Titans and their enemies across the ocean. This turns a simple battle for survival into a morally ambiguous global conflict.")

print(result)
print(result['summary'])
print(result['sentiment'])