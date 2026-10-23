from abc import ABC, abstractmethod


class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass


import random
class DummyLLM(Runnable):

    def __init__(self):
        print('LLM Created')

    def invoke(self, prompt):
        response_list = [
            'Response 1',
            'Response 2',
            'Response 3'
        ]

        return {'response': random.choice(response_list)}
    

    def predict(self, prompt):
        response_list = [
            'Response 1',
            'Response 2',
            'Response 3'
        ]

        return {'response': random.choice(response_list)}
    


class DummyPromptTemplate(Runnable):

        def __init__(self, template, input_variables):
            self.template = template
            self.input_variavles = input_variables

        def invoke(self, input_dict):
            return self.template.format(**input_dict)
        
        def format(self, input_dict):
            return self.template.format(**input_dict)
        
    

class DummyStrOutputParser(Runnable):

        def __init__(self):
            pass

        def invoke(self, input_data):
            return input_data['response']
        


class RunnableConnector(Runnable):

        def __init__(self, runnable_list):
            self.runnable_list = runnable_list

        def invoke(self, input_data):
            for runnable in self.runnable_list:
                input_data = runnable.invoke(input_data)

                return input_data



        
            
template = DummyPromptTemplate(
        template='Anime is {what} but {anime} hits different',
        input_variables=['what', 'anime']
    )


llm = DummyLLM()

parser = DummyStrOutputParser()
chain = RunnableConnector([template, llm, parser])

print(chain.invoke({'what':'Dark', 'anime':'Tokyo Ghoul'}))


template1 = DummyPromptTemplate(
    template='Write a joke about {animeT1}',
    input_variables=['animeT1']
)

template2 = DummyPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

llm = DummyLLM()

parser = DummyStrOutputParser()

chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])

finalChain = RunnableConnector([chain1, chain2])
fresult = finalChain.invoke({'animeT1':'AOT'})
print(fresult)