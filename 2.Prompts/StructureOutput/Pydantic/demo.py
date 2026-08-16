from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'John Doe'
    age: int = 32
    job: Optional[str] = "Jobless"

    #build in Validation
    email: EmailStr

    #Custom Validation
    cgpa: float = Field(..., gt=0.0, lt=10.0, description="CGPA must be between 0.0 and 10.0") 



newS = {"name": "John Doe", 'email':'xyz@gmail.com', 'cgpa': 8.5} #Only string values are allowed for the name field, so this will raise a validation error if a non-string value is provided.

student = Student(**newS)
print(student)
print(student.age)
print(type(student))