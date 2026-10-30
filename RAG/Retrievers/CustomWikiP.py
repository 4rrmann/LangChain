import requests
from langchain_core.documents import Document

def wikipedia_retriever(query, top_k=2):

    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AOTLearningProject/1.0 (educational use)"
    }

    # Search Wikipedia
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json"
    }

    response = requests.get(
        url,
        params=search_params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    results = response.json()["query"]["search"]

    docs = []

    # Fetch full article extracts
    for result in results:

        pageid = result["pageid"]

        extract_params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "pageids": pageid,
            "format": "json"
        }

        response = requests.get(
            url,
            params=extract_params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        page = response.json()["query"]["pages"][str(pageid)]

        docs.append(
            Document(
                page_content=page.get("extract", ""),
                metadata={
                    "title": page["title"],
                    "pageid": pageid
                }
            )
        )

    return docs


# Query
query = "Attack on Titan main characters"

# Retrieve documents
docs = wikipedia_retriever(query, top_k=2)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 60)