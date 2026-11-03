from langchain_core.tools import tool

'''STEPS TO CREATE A CUSTOM TOOL With @tool Decorator'''

# 1. Create a function
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

# 2. Add type hints
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

# 3. Add tool decorator
@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a*b



result = multiply.invoke({"a":1, "b":6})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)


print("\nwhat LLM sees in it:\n",multiply.args_schema.model_json_schema())





'''USING STRUCTURE TOOL & PYDANTIC (uses a Pydantic BaseModel to automatically type-check, validate, and document a function's schema so LLMs can accurately call complex APIs with multiple arguments)'''

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

def multiply(a: int, b:int) -> int:
    return a*b

multiplytool = StructuredTool.from_function(
    func=multiply,
    name="Multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiplytool.invoke({'a':2, 'b':3})
print(result)
print(multiplytool.name)
print(multiplytool.description)





'''USING BASETOOL (abstract base class for all tools in LangChain. It defines the core structure and interface that any tool must follow, whether it's a simple one-liner or a fully customized function)'''

from langchain_core.tools import BaseTool
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a*b
    

multiplytool = MultiplyTool()

result = multiplytool.invoke({'a':6, 'b':8})
print(result)

print(multiplytool.name)
print(multiplytool.description)

print(multiplytool.args)