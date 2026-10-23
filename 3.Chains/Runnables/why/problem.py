import random

# Component: 1
class DummyLLM:

    def __init__(self):
        print('LLM created')

    def predict(self, prompt):
        response_list = [
            'Response1',
            'Response2',
            'Response3',
        ]

        return {'response': random.choice(response_list)}
    


llm = DummyLLM()
print(llm.predict('query'))



# Component: 2
class DummpyPromptTemplate:

    def __init__(self, template, input_var):
        self.template = template
        self.input_var = input_var

    def Format(self, input_dict):
        return self.template.format(**input_dict)
    


template = DummpyPromptTemplate(
    template='Anime is {about} but {name} hits different.` ',
    input_var=['about', 'name']
)

prompt = template.Format({'about':'cool', 'name':'AOT'})
print(f'\n{prompt}\n')

print(llm.predict(prompt))



#
print('\nDummyLLMChain:')
class DummyLLMChain:

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):

        final_prompt = self.prompt.Format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']


template = DummpyPromptTemplate(
    template='Anime is {about} but {name} hits different.` ',
    input_var=['about', 'name']
)


llm = DummyLLM()


chain = DummyLLMChain(llm, template)


Foutput = chain.run({'about':'Rage', 'name':'Death Note'})
print(Foutput)