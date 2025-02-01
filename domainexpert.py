from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

deepseek_groq=ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key="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",
    temperature=0.7,
    max_tokens=1028,
    verbose=True
)


template = """
You are a domain expert agent specializing in dataset analysis. Your task is to classify columns of a dataset into:
1. Target Variable
2. Useful Variables
3. Not Useful Variables

Inputs:
- Dataset Summary: {dataset_summary}
- Correlation Statistics: {correlation_stats}

Output your classification. Explain your reasoning for every classification.

Begin:
"""
prompt = PromptTemplate(input_variables=["dataset_summary", "correlation_stats"], template=template)


parser = StrOutputParser()

dsgroq_chain = prompt |deepseek_groq | parser
