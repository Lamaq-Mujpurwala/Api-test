from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd


llm = ChatGroq(
    api_key="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",
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=2048,
    verbose=True,
)

prompt_template = """
You are an advanced data analysis assistant. Your task is to help the user analyze a dataset, determine whether specific user-defined data science or machine learning concepts can be learned from the dataset, and create a step-by-step plan to learn these concepts.

### Instructions:
1. **Dataset Analysis**:
   - Examine the dataset provided (columns, data types, and a sample of the data).
   - Identify patterns, correlations, or anomalies in the data.
   - Summarize the dataset's key characteristics, including its structure, data types, missing values, and overall distribution.

2. **Assess Learning Feasibility**:
   - Accept a user-defined concept (e.g., Linear Regression, Exploratory Data Analysis, Clustering, etc.).
   - Determine if the dataset contains relevant features or patterns to learn and apply the concept.
   - Justify whether the concept is applicable and feasible or provide reasons why it may not be feasible.

3. **Devise a Learning Plan**:
   - If the concept is feasible, provide a detailed step-by-step plan to apply it.
   - Include techniques such as data preprocessing, feature engineering, model selection, and evaluation methods relevant to the concept.
   - Suggest appropriate visualizations, metrics, or approaches to validate the results.

### Example:
User-defined Concept: "Linear Regression."
Dataset: Contains columns for years of experience and salary.

Your response should be detailed, structured, and actionable.

User input : 

Concept : {concept}
Dataset : {dataset}
"""




dataset_prompt=PromptTemplate(
    template=prompt_template,
    input_variables=["dataset","concept"]
)

parser = StrOutputParser()

dataset_expert_chain= dataset_prompt | llm | parser
