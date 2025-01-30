from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
groq_api_key = "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"

# Set up the LangChain LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=1028,
    max_retries=2,
    verbose=True,
)

# Define the prompt template
prompt_template = """
You are an expert Python data visualization developer. Your task is to generate Python code for a clean and functional data visualization based on the following input:

Features: {features}
Dataset summary: {dataset_summary}
Visualization type: {visualization_type}

Requirements:
1. The code must assume the dataset is already loaded into a variable called 'dataset'.
2. Ensure the code uses matplotlib or seaborn.
3. Handle edge cases like missing columns or empty datasets with error checks.
4. Return only the Python code block that can be executed directly without any additional modifications.
5. Do not include any additional text, comments, or triple backticks in your response.
"""

prompt = PromptTemplate(input_variables=["features", "dataset_summary", "visualization_type"], template=prompt_template)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser