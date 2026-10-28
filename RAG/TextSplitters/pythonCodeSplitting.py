from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

aot_code = '''
class Titan:
    def __init__(self, name, height_m):
        self.name = name
        self.height_m = height_m

    def roar(self):
        return f"{self.name} roars!"


class Soldier:
    def __init__(self, name, squad):
        self.name = name
        self.squad = squad

    def attack(self, titan):
        return f"{self.name} of {self.squad} slices the nape of {titan.name}!"


def battle():
    eren = Soldier("Eren", "Survey Corps")
    colossal = Titan("Colossal Titan", 60)
    print(colossal.roar())
    print(eren.attack(colossal))


if __name__ == "__main__":
    battle()
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=200, chunk_overlap=0
)

chunks = splitter.split_text(aot_code)
print("No. of Chunks: ",len(chunks))

for i, chunk in enumerate(splitter.create_documents([aot_code]), 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(chunk.page_content)