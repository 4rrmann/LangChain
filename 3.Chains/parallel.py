from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGroq(
    model='openai/gpt-oss-120b'
)

model2 = ChatGroq(
    model='qwen/qwen3.6-27b'
)

model3 = ChatGroq(
    model='openai/gpt-oss-20b'
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n Notes: {notes} \n Quiz: {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chian = prompt3 | model3 | parser

chain = parallel_chain | merge_chian

text = """
Attack on Titan (Shingeki no Kyojin), created by Hajime Isayama, is a dark fantasy, action, political, and philosophical anime that begins as a seemingly straightforward story about humanity fighting giant humanoid creatures called Titans but gradually transforms into a complex narrative about freedom, war, oppression, nationalism, morality, hatred, inherited trauma, and the consequences of human choices. The story takes place in a world where the remaining human population lives inside three enormous concentric walls—Wall Maria, Wall Rose, and Wall Sina—to protect themselves from Titans, mysterious giant beings that devour humans without any obvious reason. The protagonist, Eren Yeager, lives in the Shiganshina District with his adoptive sister Mikasa Ackerman and best friend Armin Arlert. Eren desperately wants to see the world beyond the Walls and hates the restrictions imposed on humanity. His worldview changes when the Colossal Titan suddenly appears and destroys part of Wall Maria, while the Armored Titan breaks through another section, allowing countless Titans to enter. Eren witnesses his mother Carla being eaten by a Titan, creating an intense desire for revenge. Eren, Mikasa, and Armin eventually join the military, where they train alongside characters such as Jean Kirstein, Connie Springer, Sasha Blouse, Historia Reiss, Reiner Braun, Bertolt Hoover, Annie Leonhart, and others. Initially, the Survey Corps appears to be humanity's primary hope because they venture outside the Walls to investigate Titans and discover the truth about the world. Eren possesses a mysterious ability to transform into a Titan, initially shocking everyone because Titans were believed to be humanity's enemies. His Titan power becomes an important weapon against other Titans, but the real significance of his ability is revealed gradually. One of the earliest major mysteries is the identity of the Female Titan, eventually revealed to be Annie Leonhart, who is secretly an enemy infiltrator from outside the Walls. Later, Reiner Braun reveals that he is the Armored Titan and Bertolt is the Colossal Titan, exposing a devastating truth: several people Eren trusted were actually warriors sent to infiltrate the Walls and obtain the Founding Titan. Their actions caused enormous destruction, but AOT deliberately complicates the concept of good and evil by showing that Reiner, Bertolt, and Annie were themselves products of a militaristic society and were raised to believe that the people inside the Walls were dangerous enemies. The story then reveals that humanity inside the Walls is not the last surviving civilization. Instead, the people of Paradis Island have been isolated from the rest of the world, while a technologically advanced nation called Marley controls much of the world and discriminates against Eldians, people descended from Ymir Fritz who possess the ability to become Titans. Eldians living in Marley are forced into segregated zones and are treated as inferior because of the historical atrocities associated with their ancestors. Marley trains selected Eldian children as Warrior candidates and uses them as weapons, granting them powerful Titan abilities. This introduces one of AOT's central themes: people inherit hatred and guilt for actions they personally never committed. The history of Titans begins with Ymir Fritz, an enslaved girl who obtains the mysterious power of the Titans roughly two thousand years before the main story. After her death, her power is divided among the Nine Titans, including the Founding Titan, Attack Titan, Colossal Titan, Armored Titan, Female Titan, Beast Titan, Jaw Titan, Cart Titan, and War Hammer Titan. The Founding Titan possesses extraordinary abilities over Titans and Eldians, including the ability to manipulate memories and biological characteristics, while the Attack Titan possesses a unique ability connected to memories of future inheritors. Eren eventually inherits both the Attack Titan and Founding Titan, making him one of the most powerful individuals in the story. His half-brother Zeke Yeager, son of Grisha Yeager and Dina Fritz, inherits the Beast Titan and develops his own ideology for ending the suffering of Eldians. Zeke proposes the Eldian euthanasia plan, intending to prevent future generations of Eldians from being born so that the Titan curse and persecution will eventually disappear. Eren initially appears to cooperate with Zeke, but his actual goal is radically different. Through the Paths, a mysterious metaphysical dimension connecting all Eldians and Titan powers, Eren reaches Ymir and gains access to the Founding Titan's full power. He eventually initiates the Rumbling, unleashing countless enormous Wall Titans that march across the world and destroy civilizations. Eren claims that he is doing this to protect Paradis and ensure the freedom of his friends, but the Rumbling becomes one of the darkest moral questions in the entire series because protecting one population requires the mass killing of millions of innocent people. Eren's character therefore changes dramatically from an angry boy who simply wanted to kill Titans into someone willing to become the world's greatest threat in pursuit of his interpretation of freedom. The Attack Titan's relationship with time and memory further complicates his character because Eren can access memories connected to future inheritors, creating a disturbing relationship between cause and effect. His father, Grisha, originally steals the Founding Titan from the royal Reiss family, and later memories reveal that Eren influenced Grisha through the Paths, demonstrating how the future Eren could affect decisions in the past. This creates the impression that Eren is simultaneously a victim of fate and an architect of it. Armin, Mikasa, Levi, Reiner, Jean, Connie, Historia, and others eventually form an alliance despite previously being enemies, because they recognize that Eren's Rumbling must be stopped. Mikasa's relationship with Eren becomes especially important because she loves him deeply but ultimately understands that stopping him requires killing him. Levi represents another important perspective on duty and sacrifice, repeatedly losing comrades while continuing to fight. Historia Reiss, meanwhile, becomes important to the political transformation of Paradis and demonstrates how individuals can reject identities imposed upon them. AOT also explores political corruption through the government inside the Walls, showing that the monarchy has deliberately manipulated history and erased people's memories to maintain control. The Survey Corps' discovery of Grisha's basement finally reveals the truth about the outside world and completely changes the meaning of the story. The basement contains evidence proving that humanity was never extinct beyond the Walls. From that point onward, Titans are no longer simply monsters; they become weapons, symbols of historical trauma, and tools of political power. The series repeatedly emphasizes that every character has a perspective shaped by their circumstances. Marley sees Paradis as a dangerous enemy, Paradis sees Marley as an oppressive force, Eldians experience discrimination because of their ancestry, and ordinary civilians on every side become victims of decisions made by governments and soldiers. This moral complexity is one of AOT's greatest strengths because the story rarely provides a simple hero-versus-villain structure. The concept of freedom is arguably the central theme. Eren begins by believing freedom means reaching the world beyond the Walls, but as he discovers the reality of that world, his definition becomes increasingly extreme. Armin views freedom through exploration and understanding, while Mikasa associates it with protecting the people she loves. Zeke believes freedom from suffering requires ending Eldian reproduction, while other characters believe freedom requires coexistence despite historical hatred. The series therefore asks whether freedom is possible when people are trapped by history, fear, genetics, political systems, and the consequences of previous generations. Another major theme is the cycle of violence: one generation commits violence against another, the victims grow up seeking revenge, and the next generation inherits the same hatred. AOT repeatedly demonstrates that revenge rarely creates genuine freedom because violence creates new victims who eventually become perpetrators themselves. The ending reinforces this ambiguity rather than presenting an easy solution. Eren is ultimately stopped by his friends, and the Titan curse is finally brought toward its conclusion when Mikasa kills Eren and Ymir is released from her ancient emotional attachment to King Fritz. The remaining characters survive, but the world does not magically become peaceful. The ending suggests that conflict is deeply rooted in human society and cannot simply be erased by defeating one enemy. Paradis continues preparing for war, while the surviving characters attempt to build a future through diplomacy and understanding. AOT's final message is therefore not simply that humanity should choose peace, but that peace requires confronting hatred, historical manipulation, fear, and the temptation to dehumanize an enemy. Ultimately, Attack on Titan is a story about freedom and its cost, the inherited consequences of history, the psychological effects of war, the danger of nationalism and propaganda, the ambiguity of morality, and the tragic difficulty of breaking cycles of violence. Its greatest narrative achievement is how it continuously changes the audience's understanding of its own world: Titans become humans, enemies become victims, heroes become morally questionable, and the simple desire for freedom evolves into a philosophical question about whether absolute freedom can exist without destroying someone else's freedom.
"""

result = chain.invoke({'text': text})
print(result)



#for Visualizing the Chain:
print(f"\nVISUALISING THE CHAIN:\n")
chain.get_graph().print_ascii()