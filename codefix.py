from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define FastAPI app

# Define input model for API
class CodeInput(BaseModel):
    code: str

# Define the LLM
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7,
    streaming=True  # Enables streaming
)

# Define the prompt template
coder_template = """
    You are a Senior Data Analyst and Machine Learning Engineer. Your job is to analyse any given code and ensure that it doesn't contain any errors.
    If the code is correct you will provide a short and concise report on how it works.
    Do not improve the code if not necessary.
    If the code is incorrect you must fix the code and help the user understand where they went wrong with chain of thought and logical reasoning.
    For explaining any concept use simple and clear language and avoid using jargon.
    When explaining a concept provide a non-technical analogy to help the user understand the concept better.

    Here is the code to analyze:

    Code : {code}
"""
code_prompt = PromptTemplate(
    template=coder_template,
    input_variables=["code"]
)

# Define the output parser
output_parser = StrOutputParser()

code_fix_chain = code_prompt | llm | output_parser

